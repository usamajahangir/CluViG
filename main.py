import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import pybullet as p
import pybullet_data
import time
import xml.etree.ElementTree as ET
import os

# --- Preprocessing ---
def preprocess_rgbd(rgb_path, depth_path):
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Load RGB image
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise ValueError(f"Failed to load RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Load depth image
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise ValueError(f"Failed to load depth image: {depth_path}")
    
    print(f"Original RGB shape: {rgb.shape}")
    print(f"Original depth shape: {depth.shape}")
    
    # Resize both RGB and depth to 224x224 BEFORE blending
    rgb_resized = cv2.resize(rgb, (224, 224))
    depth_resized = cv2.resize(depth, (224, 224))
    
    # Normalize depth to [0, 255]
    depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-6) * 255
    depth_normalized = depth_normalized.astype(np.uint8)
    
    print(f"Resized RGB shape: {rgb_resized.shape}")
    print(f"Resized depth shape: {depth_normalized.shape}")
    
    # Blend depth into RGB (use depth as red channel, keep green/blue)
    rgbd = rgb_resized.copy()
    rgbd[:, :, 0] = depth_normalized  # Replace red channel with depth
    
    # Convert to PIL and process
    rgbd_pil = Image.fromarray(rgbd)
    inputs = processor(images=rgbd_pil, return_tensors="pt")
    return inputs['pixel_values']

def preprocess_rgbd_from_arrays(rgb_array, depth_array):
    """Process RGB and depth arrays directly without saving to files"""
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    print(f"Input RGB shape: {rgb_array.shape}")
    print(f"Input depth shape: {depth_array.shape}")
    
    # Resize both RGB and depth to 224x224
    rgb_resized = cv2.resize(rgb_array, (224, 224))
    depth_resized = cv2.resize(depth_array, (224, 224))
    
    # Normalize depth to [0, 255]
    if depth_resized.max() > depth_resized.min():
        depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255
    else:
        depth_normalized = np.zeros_like(depth_resized)
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # Blend depth into RGB (use depth as red channel)
    rgbd = rgb_resized.copy()
    rgbd[:, :, 0] = depth_normalized  # Replace red channel with depth
    
    # Convert to PIL and process
    rgbd_pil = Image.fromarray(rgbd)
    inputs = processor(images=rgbd_pil, return_tensors="pt")
    return inputs['pixel_values']

# --- Dataset ---
class GraspDataset(Dataset):
    def __init__(self, rgb_paths, depth_paths, meta_paths):
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.meta_paths = meta_paths
        assert len(rgb_paths) == len(depth_paths) == len(meta_paths), \
            "Mismatch in number of files"

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgbd = preprocess_rgbd(self.rgb_paths[idx], self.depth_paths[idx])
        meta = scipy.io.loadmat(self.meta_paths[idx])
        grasps = meta.get('grasps', np.zeros((1, 7)))  # Adjust 'grasps' key if needed
        grasp = grasps[np.argmax(grasps[:, -1])] if grasps.shape[0] > 0 else grasps[0]
        grasp_tensor = torch.tensor(grasp[:7], dtype=torch.float32)
        return rgbd.squeeze(0), grasp_tensor

# --- Model ---
class GraspViT(nn.Module):
    def __init__(self):
        super(GraspViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(768, 7)  # 768 is ViT's hidden size, 7 is (x, y, z, quaternion)

    def forward(self, x):
        outputs = self.vit(x).last_hidden_state[:, 0, :]  # CLS token
        grasp = self.fc(outputs)
        return grasp

# --- Enhanced Simulation with End-Effector Camera ---
def simulate_grasp():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load environment
    plane = p.loadURDF('plane.urdf')
    table = p.loadURDF('table/table.urdf', [0, 0, 0])
    
    # Load robot properly positioned on the table
    panda = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.625], useFixedBase=True)
    
    # Get robot info for joint control
    num_joints = p.getNumJoints(panda)
    print(f"Robot has {num_joints} joints")
    
    # Print joint info for debugging
    for i in range(num_joints):
        joint_info = p.getJointInfo(panda, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")
    
    # Set robot to a stable initial pose with good view of workspace
    initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # 7 arm joints
    for i in range(min(7, num_joints)):
        p.resetJointState(panda, i, initial_joint_positions[i])
    
    # Open the gripper initially
    if num_joints > 7:
        p.resetJointState(panda, 9, 0.04)  # Left finger
        p.resetJointState(panda, 10, 0.04)  # Right finger

    # Add colorful objects to grasp at different positions
    objects = []
    object_positions = [
        [0.39, 0.10, 0.65],   # Closer to robot
        # [0.3, -0.2, 0.65],  # Slightly to the side
        # [0.3, 0.3, 0.65]    # Another position
    ]
    
    object_colors = [
        [1, 0, 0, 1],  # Red
        # [0, 1, 0, 1],  # Green  
        # [0, 0, 1, 1]   # Blue
    ]
    
    for i, (pos, color) in enumerate(zip(object_positions, object_colors)):
        obj = p.loadURDF('cube_small.urdf', pos, [0, 0, 0, 1])
        p.changeVisualShape(obj, -1, rgbaColor=color)
        objects.append(obj)
        print(f"Loaded object {i} at position {pos}")
    
    # Let physics settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)

    def get_end_effector_camera_view():
        """Get RGB and depth images from camera mounted on end-effector"""
        # Get end-effector pose
        ee_state = p.getLinkState(panda, 11)  # End effector link
        ee_pos = np.array(ee_state[0])
        ee_orn = ee_state[1]
        
        print(f"End-effector position: {ee_pos}")
        print(f"End-effector orientation (quat): {ee_orn}")
        
        # Convert quaternion to rotation matrix
        ee_rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        
        # Camera mounted on end-effector - position it slightly forward and up from gripper
        # Adjust these offsets based on your robot's gripper design
        camera_offset_local = np.array([0.0, 0.0, 0.05])  # 5cm forward in gripper's local Z direction
        camera_pos = ee_pos + ee_rot_matrix @ camera_offset_local
        
        # Camera looks down towards the workspace
        # The target should be below the end-effector
        look_distance = 0.25  # Look 25cm ahead
        look_direction_local = np.array([0.0, 0.0, look_distance])  # Look down in local coordinates
        camera_target = ee_pos + ee_rot_matrix @ look_direction_local
        
        # Camera up vector - use the gripper's Y axis as up
        up_vector_local = np.array([0.0, 1.0, 0.0])
        camera_up = ee_rot_matrix @ up_vector_local
        
        print(f"Camera position: {camera_pos}")
        print(f"Camera target: {camera_target}")
        print(f"Camera up vector: {camera_up}")
        
        # Create view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=camera_up.tolist()
        )
        
        # Create projection matrix
        width, height = 224, 224
        fov = 60  # Field of view
        aspect = width / height
        near_plane = 0.01
        far_plane = 5.0
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near_plane,
            farVal=far_plane
        )
        
        # Get camera image
        images = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb = np.array(images[2])  # RGB array
        depth = np.array(images[3])  # Depth array
        
        print(f"Raw camera RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
        
        # Process RGB image
        if len(rgb.shape) == 1:
            if rgb.size == width * height * 4:  # RGBA
                rgb = rgb.reshape(height, width, 4)[:, :, :3]
            elif rgb.size == width * height * 3:  # RGB
                rgb = rgb.reshape(height, width, 3)
        elif len(rgb.shape) == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]  # Remove alpha channel
        
        # Process depth image
        if len(depth.shape) == 1:
            depth = depth.reshape(height, width)
        
        # Convert depth from normalized values to more useful range
        # PyBullet depth is normalized between near and far plane
        depth_real = far_plane * near_plane / (far_plane - (far_plane - near_plane) * depth)
        depth_real = np.clip(depth_real, 0, far_plane)
        
        print(f"Processed camera RGB shape: {rgb.shape}, Depth shape: {depth_real.shape}")
        print(f"Depth range: {depth_real.min():.3f} to {depth_real.max():.3f}")
        
        return rgb.astype(np.uint8), depth_real

    # Load model
    model = GraspViT()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try to load the model with different possible filenames
    model_files = ['grasp_model-v1.0.0.pth', 'grasp_model.pth']
    model_loaded = False
    
    for model_file in model_files:
        try:
            model.load_state_dict(torch.load(model_file, weights_only=True))
            print(f"Loaded model from {model_file}")
            model_loaded = True
            break
        except FileNotFoundError:
            continue
    
    if not model_loaded:
        print(f"Warning: No model file found. Tried: {model_files}")
        print("Will demonstrate camera functionality without grasp prediction")
    else:
        model.to(device)
        model.eval()

    def move_robot_to_scan_position():
        """Move robot to a good position for scanning the workspace"""
        scan_joint_positions = [0.0, -0.5, 0, -2.0, 0, 1.5, 0.785]
        
        for i in range(7):
            p.setJointMotorControl2(
                panda, i,
                p.POSITION_CONTROL,
                targetPosition=scan_joint_positions[i],
                force=500
            )
        
        # Wait for movement to complete
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1./240.)

    def get_inverse_kinematics(target_pos, target_orn=None):
        """Use PyBullet's built-in inverse kinematics"""
        if target_orn is None:
            # Default downward-facing orientation for grasping
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Use PyBullet's inverse kinematics solver
        joint_poses = p.calculateInverseKinematics(
            panda,
            11,  # End effector link index
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        
        return joint_poses[:7]  # Return only the arm joint positions

    def move_to_position(target_pos, target_orn=None, max_steps=300, distance_threshold=0.00):
        """Move robot to target position using inverse kinematics"""
        print(f"Moving to position: {target_pos}")
        
        # Get target joint positions using IK
        target_joints = get_inverse_kinematics(target_pos, target_orn)
        print(f"Target joint positions: {target_joints}")
        
        # Move joints to target positions
        for i in range(7):
            p.setJointMotorControl2(
                panda, i,
                p.POSITION_CONTROL,
                targetPosition=target_joints[i],
                force=500,
                maxVelocity=2.0,
            )
        
        # Wait for movement to complete
        for step in range(500):
            # Check if we've reached the target
            print("waiting...{}".format(step))
            current_pos = p.getLinkState(panda, 11)[0]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            
            if distance < distance_threshold:  # Within 2cm
                print(f"Reached target position in {step} steps. Distance: {distance:.3f}m")
                break
            
            p.stepSimulation()
            time.sleep(1./240.)
        
        print(f"Distance to target: {distance:.3f}m")
        return distance < distance_threshold

    def find_closest_object():
        """Find the closest object to grasp"""
        ee_state = p.getLinkState(panda, 11)
        ee_pos = np.array(ee_state[0])
        
        closest_obj = None
        min_distance = float('inf')
        
        for i, obj in enumerate(objects):
            obj_pos = np.array(p.getBasePositionAndOrientation(obj)[0])
            distance = np.linalg.norm(ee_pos - obj_pos)
            
            if distance < min_distance:
                min_distance = distance
                closest_obj = obj
        
        if closest_obj is not None:
            obj_pos = np.array(p.getBasePositionAndOrientation(closest_obj)[0])
            print(f"Closest object at position: {obj_pos}, distance: {min_distance:.3f}m")
            return obj_pos
        
        return None

    def execute_grasp(grasp_prediction=None):
        """Execute grasping motion"""
        
        if grasp_prediction is not None:
            print(f"Using model prediction: {grasp_prediction}")
            # Convert model prediction to world coordinates
            # Assuming the model outputs relative coordinates in range [-1, 1]
            workspace_center = np.array([0.4, 0.0, 0.65])
            workspace_scale = np.array([0.2, 0.2, 0.05])  # Reduced scale
            
            target_pos = workspace_center + grasp_prediction[:3] * workspace_scale
            target_pos[2] = max(target_pos[2], 0.63)  # Don't go below table
            
            # Use predicted quaternion or default downward orientation
            if len(grasp_prediction) >= 7:
                target_orn = grasp_prediction[3:7]
                # Normalize quaternion
                target_orn = target_orn / np.linalg.norm(target_orn)
            else:
                target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        else:
            # Fallback: find closest object
            print("No model prediction, finding closest object...")
            target_pos = find_closest_object()
            if target_pos is None:
                print("No objects found!")
                return False
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])  # Downward facing
        
        print(f"Target grasp position: {target_pos}")
        
        # Step 1: Move to pre-grasp position (above target)
        pre_grasp_pos = target_pos.copy()
        print(type(pre_grasp_pos))
        print(f"Pre-grasp position: {pre_grasp_pos}")
        pre_grasp_pos[2] += 0.1  # 10cm above target
        # return
        
        print("Moving to pre-grasp position...")
        if not move_to_position(pre_grasp_pos, target_orn, distance_threshold=0.01):
            print("Failed to reach pre-grasp position")
            return False
        
        # Step 2: Open gripper
        print("Opening gripper...")
        p.setJointMotorControl2(panda, 9, p.POSITION_CONTROL, targetPosition=0.04, force=100)
        p.setJointMotorControl2(panda, 10, p.POSITION_CONTROL, targetPosition=0.04, force=100)
        
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Step 3: Move down to grasp position
        print("Moving to grasp position...")
        if not move_to_position(target_pos, target_orn, max_steps=150, distance_threshold=0.01):
            print("Failed to reach grasp position")
            return False
        
        # Step 4: Close gripper
        print("Closing gripper...")
        p.setJointMotorControl2(panda, 9, p.POSITION_CONTROL, targetPosition=0.0, force=100)
        p.setJointMotorControl2(panda, 10, p.POSITION_CONTROL, targetPosition=0.0, force=100)
        
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Step 5: Lift object
        print("Lifting object...")
        lift_pos = target_pos.copy()
        lift_pos[2] += 0.15  # Lift 15cm
        
        if not move_to_position(lift_pos, target_orn, distance_threshold=0.01):
            print("Failed to lift object")
            return False
        
        print("Grasp execution completed!")
        return True

    try:
        print("Moving robot to scanning position...")
        move_robot_to_scan_position()
        
        print("Getting camera feed from end-effector...")
        rgb, depth = get_end_effector_camera_view()
        
        # Save images for inspection
        cv2.imwrite('ee_camera_rgb.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite('ee_camera_depth.png', (depth / depth.max() * 255).astype(np.uint8))
        print("Saved camera images: ee_camera_rgb.png, ee_camera_depth.png")
        
        grasp_prediction = None
        
        if model_loaded:
            print("Processing image through grasp prediction model...")
            rgbd = preprocess_rgbd_from_arrays(rgb, depth)
            rgbd = rgbd.to(device)
            
            with torch.no_grad():
                grasp_prediction = model(rgbd).cpu().numpy()[0]  # Remove batch dimension
            
            print("Predicted grasp:", grasp_prediction)
        else:
            print("No model loaded - will grasp closest object")
        
        # Execute the grasp
        success = execute_grasp(grasp_prediction)
        
        if success:
            print("Grasp successful!")
        else:
            print("Grasp failed!")
        
        # Keep simulation running for observation
        print("Simulation running. Press Ctrl+C or close window to exit.")
        
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1./60.)  # 60 FPS
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
            
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if p.isConnected():
            p.disconnect()

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp Prediction Pipeline")
    parser.add_argument('--task', choices=['explore', 'train', 'simulate', 'evaluate', 'check_keys'],
                        default='simulate', help="Task to execute")
    args = parser.parse_args()

    if args.task == 'simulate':
        simulate_grasp()