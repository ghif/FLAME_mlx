
import os
import pickle
import warnings
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Suppress NumPy 2.x VisibleDeprecationWarning from loading old pickle files
try:
    from numpy import VisibleDeprecationWarning
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass

# Additional robust filter for the specific message
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")
from .ops import lbs, vertices2landmarks, batch_rodrigues
from .utils import Struct, to_mx, rot_mat_to_euler

class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self, config):
        super().__init__()
        print("creating the FLAME Decoder in MLX")
        with open(config.flame_model_path, "rb") as f:
            self.flame_model = Struct(**pickle.load(f, encoding="latin1"))
        
        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.use_face_contour = config.use_face_contour
        self.faces = to_mx(self.flame_model.f, dtype=mx.int32)
        
        # Shape betas
        default_shape = mx.zeros((self.batch_size, 300 - config.shape_params))
        self.shape_betas = default_shape # MLX doesn't have register_parameter, just use arrays or state
        
        # Expression betas
        default_exp = mx.zeros((self.batch_size, 100 - config.expression_params))
        self.expression_betas = default_exp
        
        # Eyeball and neck rotation
        self.eye_pose = mx.zeros((self.batch_size, 6))
        self.neck_pose = mx.zeros((self.batch_size, 3))
        self.transl = mx.zeros((self.batch_size, 3))
        
        self.use_3D_translation = config.use_3D_translation
        
        # Template vertices
        self.v_template = to_mx(self.flame_model.v_template)
        
        # Shape components
        self.shapedirs = to_mx(self.flame_model.shapedirs)
        
        # J regressor
        self.J_regressor = to_mx(self.flame_model.J_regressor)
        
        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.posedirs = to_mx(posedirs)
        
        # Parent indices
        parents = to_mx(self.flame_model.kintree_table[0], dtype=mx.int32)
        # parents[0] = -1 in PyTorch, but MLX index can't be -1 for some ops.
        # Actually, lbs handles parents[1:] so it's fine.
        self.parents = parents
        
        # LBS weights
        self.lbs_weights = to_mx(self.flame_model.weights)
        
        # Static landmarks
        if os.path.exists(config.static_landmark_embedding_path):
            with open(config.static_landmark_embedding_path, "rb") as f:
                static_embeddings = Struct(**pickle.load(f, encoding="latin1"))
            
            self.lmk_faces_idx = to_mx(static_embeddings.lmk_face_idx, dtype=mx.int32)
            self.lmk_bary_coords = to_mx(static_embeddings.lmk_b_coords)
        else:
            print(f"Warning: Static landmark embedding path {config.static_landmark_embedding_path} not found. Landmarks will not be computed.")
            self.lmk_faces_idx = None
            self.lmk_bary_coords = None
        
        if self.use_face_contour and os.path.exists(config.dynamic_landmark_embedding_path):
            contour_embeddings = np.load(
                config.dynamic_landmark_embedding_path,
                allow_pickle=True,
                encoding="latin1",
            )
            contour_embeddings = contour_embeddings[()]
            self.dynamic_lmk_faces_idx = to_mx(contour_embeddings["lmk_face_idx"], dtype=mx.int32)
            self.dynamic_lmk_bary_coords = to_mx(contour_embeddings["lmk_b_coords"])
            
            neck_kin_chain = []
            curr_idx = self.NECK_IDX
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                # kintree_table[0] has parents
                curr_idx = int(self.flame_model.kintree_table[0][curr_idx]) if curr_idx != 0 else -1
            self.neck_kin_chain = mx.array(neck_kin_chain, dtype=mx.int32)
        else:
            if self.use_face_contour:
                print(f"Warning: Dynamic landmark embedding path {config.dynamic_landmark_embedding_path} not found. Face contour will not be computed.")
            self.use_face_contour = False
            self.dynamic_lmk_faces_idx = None
            self.dynamic_lmk_bary_coords = None

    def _find_dynamic_lmk_idx_and_bcoords(
        self,
        vertices,
        pose,
        dynamic_lmk_faces_idx,
        dynamic_lmk_b_coords,
        neck_kin_chain,
    ):
        """
        Selects the face contour depending on the relative position of the head
        """
        batch_size = vertices.shape[0]
        
        # pose is (B, J, 3) where J is number of joints
        # Index select across joints dimension
        aa_pose = pose[:, neck_kin_chain] # (B, K, 3)
        
        # batch_rodrigues expects (N, 3)
        rot_mats = batch_rodrigues(aa_pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
        
        rel_rot_mat = mx.broadcast_to(mx.eye(3)[None], (batch_size, 3, 3))
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = mx.matmul(rot_mats[:, idx], rel_rot_mat)
            
        # y_rot_angle calculation
        y_rot_angle_rad = rot_mat_to_euler(rel_rot_mat)
        y_rot_angle = mx.round(mx.clip(-y_rot_angle_rad * 180.0 / np.pi, a_min=-100, a_max=39)).astype(mx.int32)
        
        # Handle negative angles (same logic as PyTorch)
        neg_mask = y_rot_angle < 0
        mask = y_rot_angle < -39
        neg_vals = mask.astype(mx.int32) * 78 + (1 - mask.astype(mx.int32)) * (39 - y_rot_angle)
        y_rot_angle = mx.where(neg_mask, neg_vals, y_rot_angle)
        
        dyn_lmk_faces_idx = dynamic_lmk_faces_idx[y_rot_angle]
        dyn_lmk_b_coords = dynamic_lmk_b_coords[y_rot_angle]
        
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def __call__(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
        neck_pose=None,
        eye_pose=None,
        transl=None,
    ):
        """
        Input:
            shape_params: B X number of shape parameters
            expression_params: B X number of expression parameters
            pose_params: B X 6 (global rotation + jaw)
        """
        if shape_params is None:
            shape_params = mx.zeros((self.batch_size, 100)) # Default 100
        if expression_params is None:
            expression_params = mx.zeros((self.batch_size, 50)) # Default 50
        if pose_params is None:
            pose_params = mx.zeros((self.batch_size, 6))

        betas = mx.concatenate(
            [shape_params, self.shape_betas, expression_params, self.expression_betas],
            axis=1,
        )
        
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose
        eye_pose = eye_pose if eye_pose is not None else self.eye_pose
        transl = transl if transl is not None else self.transl
        
        # pose_params[:, :3] is global rotation, pose_params[:, 3:] is jaw
        # full_pose: [global, neck, jaw, eyes]
        full_pose = mx.concatenate(
            [pose_params[:, :3], neck_pose, pose_params[:, 3:6], eye_pose], axis=1
        ).reshape(self.batch_size, -1, 3)
        
        template_vertices = mx.broadcast_to(self.v_template[None], (self.batch_size, *self.v_template.shape))
        
        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=True
        )
        
        if self.lmk_faces_idx is not None:
            lmk_faces_idx = mx.broadcast_to(self.lmk_faces_idx[None], (self.batch_size, *self.lmk_faces_idx.shape))
            lmk_bary_coords = mx.broadcast_to(self.lmk_bary_coords[None], (self.batch_size, *self.lmk_bary_coords.shape))
            
            if self.use_face_contour:
                dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                    vertices,
                    full_pose,
                    self.dynamic_lmk_faces_idx,
                    self.dynamic_lmk_bary_coords,
                    self.neck_kin_chain
                )
                lmk_faces_idx = mx.concatenate([dyn_lmk_faces_idx, lmk_faces_idx], axis=1)
                lmk_bary_coords = mx.concatenate([dyn_lmk_bary_coords, lmk_bary_coords], axis=1)
                
            landmarks = vertices2landmarks(vertices, self.faces, lmk_faces_idx, lmk_bary_coords)
        else:
            landmarks = None
        
        if self.use_3D_translation:
            if landmarks is not None:
                landmarks += transl[:, None, :]
            vertices += transl[:, None, :]
            
        return vertices, landmarks
