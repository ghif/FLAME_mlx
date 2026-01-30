
import mlx.core as mx
from typing import Tuple, List
from .utils import rot_mat_to_euler

def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    """
    Calculates landmarks by barycentric interpolation
    """
    batch_size, num_verts = vertices.shape[:2]
    
    # Extract the indices of the vertices for each face
    # lmk_faces_idx is BxL
    # faces is Fx3
    
    # Flatten everything to select indices
    # lmk_faces is BxLx3
    lmk_faces = faces[lmk_faces_idx]
    
    # Add batch offset to indices
    # vertices is (B, V, 3) -> flattened (B*V, 3)
    # lmk_faces needs to be (B, L, 3) with absolute indices
    batch_offset = mx.arange(batch_size)[:, None, None] * num_verts
    lmk_faces_absolute = lmk_faces + batch_offset
    
    # Gather vertices
    # vertices_flat is (B*V, 3)
    vertices_flat = vertices.reshape(-1, 3)
    lmk_vertices = vertices_flat[lmk_faces_absolute] # (B, L, 3, 3)
    
    # lmk_vertices: (B, L, 3, 3)
    # lmk_bary_coords: (B, L, 3)
    # Output: (B, L, 3)
    landmarks = mx.einsum('blfi,blf->bli', lmk_vertices, lmk_bary_coords)
    return landmarks

def vertices2joints(J_regressor, vertices):
    """
    Calculates the 3D joint locations from the vertices
    """
    return mx.einsum('bik,ji->bjk', vertices, J_regressor)

def blend_shapes(betas, shape_disps):
    """
    Calculates the per vertex displacement due to the blend shapes
    """
    return mx.einsum('bl,mkl->bmk', betas, shape_disps)

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    """
    Calculates the rotation matrices for a batch of rotation vectors
    """
    batch_size = rot_vecs.shape[0]
    
    angle = mx.linalg.norm(rot_vecs + epsilon, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle
    
    cos = mx.expand_dims(mx.cos(angle), axis=1) # (B, 1, 1)
    sin = mx.expand_dims(mx.sin(angle), axis=1) # (B, 1, 1)
    
    rx, ry, rz = mx.split(rot_dir, 3, axis=1)
    zeros = mx.zeros((batch_size, 1))
    
    # K matrix components: [0, -rz, ry, rz, 0, -rx, -ry, rx, 0]
    K = mx.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1).reshape(batch_size, 3, 3)
    
    ident = mx.eye(3)[None] # (1, 3, 3)
    rot_mat = ident + sin * K + (1 - cos) * mx.matmul(K, K)
    return rot_mat

def transform_mat(R, t):
    """
    Creates a batch of transformation matrices (4x4)
    """
    # R: (B, 3, 3), t: (B, 3, 1)
    # Pad R to (B, 4, 3) with zeros in last row
    # Pad t to (B, 4, 1) with 1 in last row
    B = R.shape[0]
    bottom_R = mx.zeros((B, 1, 3))
    R_homo = mx.concatenate([R, bottom_R], axis=1)
    
    bottom_t = mx.ones((B, 1, 1))
    t_homo = mx.concatenate([t, bottom_t], axis=1)
    
    return mx.concatenate([R_homo, t_homo], axis=2)

def batch_rigid_transform(rot_mats, joints, parents):
    """
    Applies a batch of rigid transformations to the joints
    """
    # rot_mats: (B, J, 3, 3)
    # joints: (B, J, 3)
    # parents: (J,)
    
    joints = joints[:, :, :, None] # (B, J, 3, 1)
    
    rel_joints = mx.array(joints)
    # parents[1:] gets parent of joint i
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    
    # Transform matrices for each joint in its own frame
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)
    
    # transform_chain: list of global transforms for each joint
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        parent_idx = int(parents[i])
        curr_res = mx.matmul(transform_chain[parent_idx], transforms_mat[:, i])
        transform_chain.append(curr_res)
    
    transforms = mx.stack(transform_chain, axis=1) # (B, J, 4, 4)
    
    # posed_joints: (B, J, 3)
    posed_joints = transforms[:, :, :3, 3]
    
    # Relativize transforms by subtracting joint position contribution
    # T' = T - [0, T*j]
    # In PyTorch: rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    
    joints_homogen = mx.concatenate([joints, mx.zeros((joints.shape[0], joints.shape[1], 1, 1))], axis=2)
    joints_homogen[:, :, 3, 0] = 0 # Handled by concatenate above but being explicit
    # Wait, the pad in PyTorch: torch.matmul(transforms, joints_homogen) is (B, J, 4, 1)
    # Then pad is [3, 0, 0, 0, 0, 0, 0, 0] which pads columns on the left? 
    # F.pad(input, pad, mode='constant', value=0) 
    # For 4D tensor: (left, right, top, bottom, front, back, ...)
    # Here it's 2D/3D? transforms is (B, J, 4, 4). joints_homogen is (B, J, 4, 1).
    # torch.matmul(transforms, joints_homogen) is (B, J, 4, 1).
    # Then pad [3, 0, 0, 0, 0, 0, 0, 0] means add 3 columns on the left of the last dim.
    # Result is (B, J, 4, 4) where first 3 columns are 0, and last column is the matmul result.
    
    transformed_joints = mx.matmul(transforms, joints_homogen) # (B, J, 4, 1)
    
    # Create the padding matrix
    B, J = transforms.shape[:2]
    padding = mx.zeros((B, J, 4, 3))
    rel_transforms_padding = mx.concatenate([padding, transformed_joints], axis=3)
    
    rel_transforms = transforms - rel_transforms_padding
    
    return posed_joints, rel_transforms

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot=True):
    """
    Performs Linear Blend Skinning
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    
    # 1. Add shape contribution
    # v_shaped = v_template + sum(beta * shapedir)
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    
    # 2. Get the joints
    J = vertices2joints(J_regressor, v_shaped)
    
    # 3. Add pose blend shapes
    ident = mx.eye(3)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
        
        # pose_feature = (rot_mats[:, 1:] - I).flatten()
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
        # pose_offsets = pose_feature @ posedirs
        pose_offsets = mx.matmul(pose_feature, posedirs).reshape(batch_size, -1, 3)
    else:
        # pose is already rot_mats (B, J, 3, 3)
        rot_mats = pose
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
        pose_offsets = mx.matmul(pose_feature, posedirs).reshape(batch_size, -1, 3)
        
    v_posed = v_shaped + pose_offsets
    
    # 4. Get the global joint location and transformation matrices
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
    
    # 5. Do skinning
    # lbs_weights is (V, J)
    # W is (B, V, J)
    W = mx.broadcast_to(lbs_weights[None], (batch_size, *lbs_weights.shape))
    
    # T = W @ A_flattened
    # A is (B, J, 4, 4) -> (B, J, 16)
    A_flat = A.reshape(batch_size, -1, 16)
    T = mx.matmul(W, A_flat).reshape(batch_size, -1, 4, 4)
    
    # Apply skinning to vertices
    v_posed_homo = mx.concatenate([v_posed, mx.ones((batch_size, v_posed.shape[1], 1))], axis=2)
    # v_homo = T @ v_posed_homo (with extra dim for matmul)
    v_homo = mx.matmul(T, v_posed_homo[:, :, :, None])
    
    verts = v_homo[:, :, :3, 0]
    
    return verts, J_transformed
