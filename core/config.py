
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="FLAME model in MLX")

    parser.add_argument(
        "--flame_model_path",
        type=str,
        default="./model/flame2023_Open.pkl",
        help="flame model path",
    )

    parser.add_argument(
        "--static_landmark_embedding_path",
        type=str,
        default="./model/flame_static_embedding.pkl",
        help="Static landmark embeddings path for FLAME",
    )

    parser.add_argument(
        "--dynamic_landmark_embedding_path",
        type=str,
        default="./model/flame_dynamic_embedding.npy",
        help="Dynamic contour embedding path for FLAME",
    )

    # FLAME hyper-parameters
    parser.add_argument(
        "--shape_params", type=int, default=100, help="the number of shape parameters"
    )

    parser.add_argument(
        "--expression_params",
        type=int,
        default=50,
        help="the number of expression parameters",
    )

    parser.add_argument(
        "--pose_params", type=int, default=6, help="the number of pose parameters"
    )

    # MLX specifics
    parser.add_argument(
        "--use_face_contour",
        default=True,
        type=bool,
        help="If true apply the landmark loss on also on the face contour.",
    )

    parser.add_argument(
        "--use_3D_translation",
        default=True,
        type=bool,
        help="If true apply the landmark loss on also on the face contour.",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")

    # Try to parse or return defaults if fail (e.g. in script)
    try:
        config = parser.parse_known_args()[0]
    except:
        config = parser.parse_args([])
        
    return config
