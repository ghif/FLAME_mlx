
import os
import mlx.core as mx
import numpy as np
from core import FLAME, get_config

def main():
    config = get_config()
    
    # Check if model files exist
    missing_files = []
    for path in [config.flame_model_path, config.static_landmark_embedding_path, config.dynamic_landmark_embedding_path]:
        if not os.path.exists(path):
            missing_files.append(path)
            
    if not os.path.exists(config.flame_model_path):
        print(f"Error: FLAME model file '{config.flame_model_path}' is missing.")
        print("Please download the FLAME model from http://flame.is.tue.mpg.de/ and place it in the correct directory.")
        return
    elif missing_files:
        print("Warning: Some landmark embedding files are missing. Landmarks will not be computed.")
        for f in missing_files:
            if f != config.flame_model_path:
                print(f"  - {f}")

    # Initialize FLAME model
    flame = FLAME(config)
    
    # Create batch of parameters
    batch_size = config.batch_size
    shape_params = mx.zeros((batch_size, 100))
    expression_params = mx.zeros((batch_size, 50))
    pose_params = mx.zeros((batch_size, 6)) # [global_rot, jaw_rot]
    
    # Forward pass
    vertices, landmarks = flame(
        shape_params=shape_params,
        expression_params=expression_params,
        pose_params=pose_params
    )
    
    print(f"Successfully ran FLAME forward pass in MLX!")
    print(f"Vertices shape: {vertices.shape}")
    if landmarks is not None:
        print(f"Landmarks shape: {landmarks.shape}")
    else:
        print("Landmarks not computed (missing embedding files).")

if __name__ == "__main__":
    main()
