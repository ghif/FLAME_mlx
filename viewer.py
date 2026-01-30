
import time
import os
import mlx.core as mx
import numpy as np
import viser
import viser.transforms as tf
from core import FLAME, get_config

def main():
    server = viser.ViserServer(port=8081,open_browser=False)
    print(f"VISER_PORT: {server.get_port()}", flush=True)
    server.scene.set_up_direction("+y")
    config = get_config()

    # Predefined model paths
    MODEL_PATHS = {
        "Generic": "./model/flame2023_Open.pkl",
        "Male": "./model/male_model.pkl",
        "Female": "./model/female_model.pkl",
    }

    # Initial state
    state = {
        "model_type": "Generic",
        "shape": np.zeros(100, dtype=np.float32),
        "expr": np.zeros(50, dtype=np.float32),
        "neck": np.zeros(3, dtype=np.float32),
        "jaw": np.zeros(3, dtype=np.float32),
    }

    # Load initial model
    def load_model(path):
        print(f"Attempting to load model from: {path}")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None
        config.flame_model_path = path
        try:
            model = FLAME(config)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    flame_model = load_model(MODEL_PATHS["Generic"])
    if flame_model is None:
        print(f"Error: Default model {MODEL_PATHS['Generic']} not found.")
        # We'll try to continue but UI might be broken
    
    # UI Components
    with server.gui.add_folder("Model Selection"):
        model_dropdown = server.gui.add_dropdown(
            "Model Type",
            options=["Generic"],
            initial_value="Generic",
        )

    shape_sliders = []
    with server.gui.add_folder("Shape Parameters"):
        for i in range(10):
            shape_sliders.append(
                server.gui.add_slider(
                    f"Shape {i}",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial_value=0.0,
                )
            )

    expr_sliders = []
    with server.gui.add_folder("Expression Parameters"):
        for i in range(10):
            expr_sliders.append(
                server.gui.add_slider(
                    f"Expression {i}",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial_value=0.0,
                )
            )

    with server.gui.add_folder("Jaw Pose"):
        jaw_x = server.gui.add_slider("Jaw Pitch", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
        jaw_y = server.gui.add_slider("Jaw Yaw", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
        jaw_z = server.gui.add_slider("Jaw Roll", min=-0.5, max=0.5, step=0.01, initial_value=0.0)

    with server.gui.add_folder("Neck Pose"):
        neck_x = server.gui.add_slider("Neck Pitch", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
        neck_y = server.gui.add_slider("Neck Yaw", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
        neck_z = server.gui.add_slider("Neck Roll", min=-0.5, max=0.5, step=0.01, initial_value=0.0)

    # Update Function
    def update_mesh():
        nonlocal flame_model
        if flame_model is None:
            return

        # Prepare parameters
        shape_mx = mx.zeros((1, 100))
        shape_vals = np.array([s.value for s in shape_sliders], dtype=np.float32)
        shape_mx[0, :10] = mx.array(shape_vals)

        expr_mx = mx.zeros((1, 50))
        expr_vals = np.array([e.value for e in expr_sliders], dtype=np.float32)
        expr_mx[0, :10] = mx.array(expr_vals)

        # Pose: [global_rot(3), jaw_rot(3)]
        # However, FLAME class expects pose_params for global and jaw, 
        # and neck/eyes as separate arguments if optimize_neckpose is True.
        # Original main.py: pose_params (B, 6) -> [global, jaw]
        
        jaw_vals = np.array([jaw_x.value, jaw_y.value, jaw_z.value], dtype=np.float32)
        global_rot = mx.zeros((1, 3)) # Keep global rotation zero for now
        pose_params = mx.concatenate([global_rot, mx.array(jaw_vals)[None]], axis=1)

        neck_vals = np.array([neck_x.value, neck_y.value, neck_z.value], dtype=np.float32)
        neck_pose = mx.array(neck_vals)[None]

        # Forward Pass
        vertices, _ = flame_model(
            shape_params=shape_mx,
            expression_params=expr_mx,
            pose_params=pose_params,
            neck_pose=neck_pose
        )

        # Convert to numpy for Viser
        verts_np = np.array(vertices[0])
        faces_np = np.array(flame_model.faces)

        # Update Viser
        server.scene.add_mesh_simple(
            "/flame_mesh",
            vertices=verts_np,
            faces=faces_np,
            color=(200, 180, 150),
            wireframe=False,
        )

    # Listeners
    @model_dropdown.on_update
    def _(_):
        nonlocal flame_model
        path = MODEL_PATHS[model_dropdown.value]
        new_model = load_model(path)
        if new_model is not None:
            flame_model = new_model
            update_mesh()
        else:
            print(f"Warning: Model file {path} not found.")

    for s in shape_sliders + expr_sliders + [jaw_x, jaw_y, jaw_z, neck_x, neck_y, neck_z]:
        s.on_update(lambda _: update_mesh())

    # Initial update
    update_mesh()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Set initial camera position for new clients
        # Position: [x, y, z], Look at: [x, y, z]
        client.camera.position = (0.0, 0.02, 0.25)
        client.camera.look_at = (0.0, 0.0, 0.0)

    print(f"Viser server started at {server.get_host()}:{server.get_port()}")
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
