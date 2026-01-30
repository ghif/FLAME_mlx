
import mlx.core as mx
from typing import Tuple, List
from .utils import rot_mat_to_euler

def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    """
    Calculates 3D landmarks by barycentric interpolation over specific mesh faces.
    
    Args:
        vertices (mx.array): Batch of mesh vertices of shape (B, V, 3).
        faces (mx.array): Mesh face indices of shape (F, 3).
        lmk_faces_idx (mx.array): Indices of faces where landmarks are located (B, L).
        lmk_bary_coords (mx.array): Barycentric coordinates of landmarks on those faces (B, L, 3).
        
    Returns:
        mx.array: Calculated 3D landmarks of shape (B, L, 3).
    """
    batch_size, num_verts = vertices.shape[:2]
    
    # Extract the indices of the vertices for each face
    lmk_faces = faces[lmk_faces_idx]
    
    # Add batch offset to indices for flat indexing
    batch_offset = mx.arange(batch_size)[:, None, None] * num_verts
    lmk_faces_absolute = lmk_faces + batch_offset
    
    # Gather vertices
    vertices_flat = vertices.reshape(-1, 3)
    lmk_vertices = vertices_flat[lmk_faces_absolute] # (B, L, 3, 3)
    
    # Compute landmarks using barycentric interpolation
    landmarks = mx.einsum('blfi,blf->bli', lmk_vertices, lmk_bary_coords)
    return landmarks

def vertices2joints(J_regressor, vertices):
    """
    Calculates 3D joint locations from mesh vertices using a regressor matrix.
    
    Args:
        J_regressor (mx.array): Joint regressor matrix of shape (J, V).
        vertices (mx.array): Batch of mesh vertices of shape (B, V, 3).
        
    Returns:
        mx.array: Calculated 3D joint locations of shape (B, J, 3).
    """
    return mx.einsum('bik,ji->bjk', vertices, J_regressor)

def blend_shapes(betas, shape_disps):
    """
    Calculates per-vertex displacements from blend shapes (identities or expressions).
    
    Args:
        betas (mx.array): Coefficients for the blend shapes of shape (B, K).
        shape_disps (mx.array): Blend shape bases of shape (V, 3, K).
        
    Returns:
        mx.array: Vertex displacements of shape (B, V, 3).
    """
    return mx.einsum('bl,mkl->bmk', betas, shape_disps)

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    """
    Converts a batch of rotation vectors (axis-angle) to 3x3 rotation matrices.
    
    Args:
        rot_vecs (mx.array): Batch of rotation vectors of shape (B, 3).
        epsilon (float): Small value to avoid division by zero.
        
    Returns:
        mx.array: Batch of 3x3 rotation matrices of shape (B, 3, 3).
    """
    batch_size = rot_vecs.shape[0]
    
    angle = mx.linalg.norm(rot_vecs + epsilon, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle
    
    cos = mx.expand_dims(mx.cos(angle), axis=1) # (B, 1, 1)
    sin = mx.expand_dims(mx.sin(angle), axis=1) # (B, 1, 1)
    
    rx, ry, rz = mx.split(rot_dir, 3, axis=1)
    zeros = mx.zeros((batch_size, 1))
    
    # Skew-symmetric matrix K
    K = mx.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1).reshape(batch_size, 3, 3)
    
    ident = mx.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * mx.matmul(K, K)
    return rot_mat

def transform_mat(R, t):
    """
    Combines rotation and translation into 4x4 homogenous transformation matrices.
    
    Args:
        R (mx.array): Batch of rotation matrices of shape (B, 3, 3).
        t (mx.array): Batch of translation vectors of shape (B, 3, 1).
        
    Returns:
        mx.array: Compiled 4x4 transformation matrices of shape (B, 4, 4).
    """
    B = R.shape[0]
    bottom_R = mx.zeros((B, 1, 3))
    R_homo = mx.concatenate([R, bottom_R], axis=1)
    
    bottom_t = mx.ones((B, 1, 1))
    t_homo = mx.concatenate([t, bottom_t], axis=1)
    
    return mx.concatenate([R_homo, t_homo], axis=2)

def batch_rigid_transform(rot_mats, joints, parents):
    """
    Computes global joint transformations based on relative rotations and kinematic chain.
    
    Args:
        rot_mats (mx.array): Batch of joint rotation matrices (B, J, 3, 3).
        joints (mx.array): Batch of joint initial positions (B, J, 3).
        parents (mx.array): Indices of parent joints for each joint (J,).
        
    Returns:
        tuple: (posed_joints, rel_transforms)
            - posed_joints (mx.array): Global joint positions after rotation (B, J, 3).
            - rel_transforms (mx.array): Relative transformation matrices for skinning (B, J, 4, 4).
    """
    joints = joints[:, :, :, None] # (B, J, 3, 1)
    
    rel_joints = mx.array(joints)
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    
    # Local transforms
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)
    
    # Global transforms via kinematic chain
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        parent_idx = int(parents[i])
        curr_res = mx.matmul(transform_chain[parent_idx], transforms_mat[:, i])
        transform_chain.append(curr_res)
    
    transforms = mx.stack(transform_chain, axis=1) # (B, J, 4, 4)
    posed_joints = transforms[:, :, :3, 3]
    
    # Compute relative transforms for skinning
    joints_homogen = mx.concatenate([joints, mx.zeros((joints.shape[0], joints.shape[1], 1, 1))], axis=2)
    transformed_joints = mx.matmul(transforms, joints_homogen) # (B, J, 4, 1)
    
    B, J = transforms.shape[:2]
    padding = mx.zeros((B, J, 4, 3))
    rel_transforms_padding = mx.concatenate([padding, transformed_joints], axis=3)
    rel_transforms = transforms - rel_transforms_padding
    
    return posed_joints, rel_transforms

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot=True):
    """
    Linear Blend Skinning (LBS) implementation.
    
    Args:
        betas (mx.array): Shape and expression coefficients (B, K).
        pose (mx.array): Pose parameters as axis-angle vectors (B, J, 3) or rotation matrices (B, J, 3, 3).
        v_template (mx.array): Template mesh vertices (B, V, 3).
        shapedirs (mx.array): Shape and expression blend shape bases (V, 3, K).
        posedirs (mx.array): Pose corrective blend shape bases (V*3, P).
        J_regressor (mx.array): Joint regressor matrix (J, V).
        parents (mx.array): Kinematic tree parent indices (J,).
        lbs_weights (mx.array): Skinning weights (V, J).
        pose2rot (bool): If True, convert 'pose' from axis-angle to rotation matrices.
        
    Returns:
        tuple: (verts, J_transformed)
            - verts (mx.array): Final posed mesh vertices of shape (B, V, 3).
            - J_transformed (mx.array): Global joint positions after transformations (B, J, 3).
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    
    # 1. Shape Blend Shapes
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    
    # 2. Get the neutral joints
    J = vertices2joints(J_regressor, v_shaped)
    
    # 3. Pose Corrective Blend Shapes
    ident = mx.eye(3)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
        pose_offsets = mx.matmul(pose_feature, posedirs).reshape(batch_size, -1, 3)
    else:
        rot_mats = pose
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
        pose_offsets = mx.matmul(pose_feature, posedirs).reshape(batch_size, -1, 3)
        
    v_posed = v_shaped + pose_offsets
    
    # 4. Global joint location and transformation matrices
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
    
    # 5. Skinning
    W = mx.broadcast_to(lbs_weights[None], (batch_size, *lbs_weights.shape))
    A_flat = A.reshape(batch_size, -1, 16)
    T = mx.matmul(W, A_flat).reshape(batch_size, -1, 4, 4)
    
    v_posed_homo = mx.concatenate([v_posed, mx.ones((batch_size, v_posed.shape[1], 1))], axis=2)
    v_homo = mx.matmul(T, v_posed_homo[:, :, :, None])
    verts = v_homo[:, :, :3, 0]
    
    return verts, J_transformed
