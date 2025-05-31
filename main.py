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

# --- Data Exploration ---
def explore_data():
    rgb_paths = sorted(glob.glob('data/train_4/scene_0090/kinect/rgb/*.png'))[:10]
    depth_paths = sorted(glob.glob('data/train_4/scene_0090/kinect/depth/*.png'))[:10]
    meta_paths = sorted(glob.glob('data/train_4/scene_0090/kinect/meta/*.mat'))[:10]
    annotations_paths = sorted(glob.glob('data/train_4/scene_0090/kinect/annotations/*.xml'))[:10]

    assert len(rgb_paths) == len(depth_paths) == len(meta_paths) == len(annotations_paths), \
        "Mismatch in number of files"

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'q' else None)

    def show_image(rgb_path, depth_path, meta_path, annotation_path):
        ax[0].clear()
        ax[1].clear()

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        meta = scipy.io.loadmat(meta_path)
        grasps = meta.get('grasps', np.zeros((1, 7)))
        grasp = grasps[np.argmax(grasps[:, -1])] if grasps.shape[0] > 0 else grasps[0]

        annotation = ET.parse(annotation_path).getroot()
        objects = []
        for obj in annotation.findall('obj'):
            obj_id = obj.find('obj_id').text
            obj_name = obj.find('obj_name').text
            pos = np.array(obj.find('pos_in_world').text.split(), dtype=float)
            ori = np.array(obj.find('ori_in_world').text.split(), dtype=float)
            objects.append({'id': obj_id, 'name': obj_name, 'pos': pos, 'ori': ori})

        ax[0].imshow(rgb)
        ax[0].set_title(f'RGB Image\nGrasp: {grasp[:3]}')
        ax[1].imshow(depth, cmap='gray')
        ax[1].set_title('Depth Map')

        print(f"\nImage: {rgb_path}")
        print("Objects:")
        for obj in objects:
            print(f"  ID: {obj['id']}, Name: {obj['name']}, Pos: {obj['pos']}, Ori: {obj['ori']}")
        print(f"Grasp (x, y, z, quaternion): {grasp}")

        plt.draw()

    index = 0
    def on_press(event):
        nonlocal index
        if event.key == 'left':
            index = (index - 1) % len(rgb_paths)
        elif event.key == 'right':
            index = (index + 1) % len(rgb_paths)
        show_image(rgb_paths[index], depth_paths[index], meta_paths[index], annotations_paths[index])

    fig.canvas.mpl_connect('key_press_event', on_press)
    show_image(rgb_paths[index], depth_paths[index], meta_paths[index], annotations_paths[index])
    plt.show()

# --- Training ---
def train_model():
    rgb_paths = sorted(glob.glob('data/train_4/*/kinect/rgb/*.png'))[:10]
    depth_paths = sorted(glob.glob('data/train_4/*/kinect/depth/*.png'))[:10]
    meta_paths = sorted(glob.glob('data/train_4/*/kinect/meta/*.mat'))[:10]
    dataset = GraspDataset(rgb_paths, depth_paths, meta_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = GraspViT()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(5):
        for rgbd, grasp in dataloader:
            rgbd, grasp = rgbd.to(device), grasp.to(device)
            optimizer.zero_grad()
            outputs = model(rgbd)
            loss = criterion(outputs, grasp)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'grasp_model.pth')

# --- Simulation ---
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
    
    # Set robot to a stable initial pose
    initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # 7 arm joints
    for i in range(min(7, num_joints)):
        p.resetJointState(panda, i, initial_joint_positions[i])
    
    # Close the gripper initially
    if num_joints > 7:
        p.resetJointState(panda, 9, 0.04)  # Left finger
        p.resetJointState(panda, 10, 0.04)  # Right finger

    # Add objects to grasp
    objects = []
    for i in range(3):
        obj = p.loadURDF('cube_small.urdf', [0.4 + i*0.15, 0.1, 0.7], [0, 0, 0, 1])
        objects.append(obj)
    
    # Let physics settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)

    def get_rgbd():
        width, height = 224, 224
        # Position camera to look at the workspace
        view_matrix = p.computeViewMatrix([0.5, 0, 1.2], [0.5, 0, 0.7], [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
        images = p.getCameraImage(width, height, view_matrix, proj_matrix)
        rgb = np.array(images[2])  # RGB is index 2
        depth = np.array(images[3])  # Depth is index 3
        
        print(f"Raw RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
        
        # Handle RGB data properly
        if len(rgb.shape) == 1:
            # If RGB is flat, reshape it
            if rgb.size == width * height * 4:  # RGBA
                rgb = rgb.reshape(height, width, 4)[:, :, :3]  # Reshape and take RGB
            elif rgb.size == width * height * 3:  # RGB
                rgb = rgb.reshape(height, width, 3)
            else:
                raise ValueError(f"Unexpected RGB size: {rgb.size}. Expected {width * height * 3} or {width * height * 4} elements.")
        elif len(rgb.shape) == 3 and rgb.shape[2] == 4:
            # If RGB has RGBA channels, take only RGB
            rgb = rgb[:, :, :3]
        
        # Handle depth data properly
        if len(depth.shape) == 1:
            depth = depth.reshape(height, width)
        
        print(f"Processed RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
        
        # Ensure proper data types
        rgb = rgb.astype(np.uint8)
        depth = (depth * 255).astype(np.uint8)  # Convert depth to 0-255 range
        
        return rgb, depth

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
        print(f"Error: No model file found. Tried: {model_files}")
        print("Please run training first with --task train")
        p.disconnect()
        return
    
    model.to(device)
    model.eval()

    try:
        rgb, depth = get_rgbd()
        cv2.imwrite('temp_rgb.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite('temp_depth.png', depth)
        rgbd = preprocess_rgbd('temp_rgb.png', 'temp_depth.png')
        rgbd = rgbd.to(device)
        with torch.no_grad():
            grasp = model(rgbd).cpu().numpy()
        print("Predicted grasp:", grasp)

        def move_to_grasp(grasp_pred):
            # Extract position and orientation from prediction
            position = grasp_pred[:3]
            quaternion = grasp_pred[3:7]
            
            print(f"Moving to grasp position: {position}")
            print(f"Grasp orientation (quaternion): {quaternion}")
            
            # Scale and adjust position to workspace
            # The predicted positions might need scaling/adjustment
            target_pos = [
                0.4 + position[0] * 0.3,  # Scale and offset X
                position[1] * 0.3,        # Scale Y  
                0.8 + position[2] * 0.2   # Scale and offset Z (above table)
            ]
            
            print(f"Adjusted target position: {target_pos}")
            
            # Move arm to pre-grasp position (slightly above target)
            pre_grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.1]
            
            # Simple inverse kinematics - move end effector towards target
            # This is a simplified approach; real IK would be more complex
            for step in range(100):
                # Get current end effector position
                ee_state = p.getLinkState(panda, 11)  # End effector link
                current_pos = ee_state[0]
                
                # Calculate direction to target
                direction = np.array(pre_grasp_pos) - np.array(current_pos)
                distance = np.linalg.norm(direction)
                
                if distance < 0.05:  # Close enough
                    break
                
                # Simple proportional control for joint movements
                # This moves joints to bring end effector closer to target
                joint_velocities = []
                for i in range(7):  # 7 arm joints
                    # Simple heuristic: adjust joints based on position error
                    if i < 3:  # First 3 joints affect position more
                        vel = direction[i] * 2.0
                    else:  # Other joints for orientation
                        vel = np.random.uniform(-0.1, 0.1)  # Small random adjustments
                    joint_velocities.append(vel)
                
                # Apply joint velocities
                p.setJointMotorControlArray(
                    panda, 
                    range(7), 
                    p.VELOCITY_CONTROL,
                    targetVelocities=joint_velocities,
                    forces=[100] * 7
                )
                
                p.stepSimulation()
                time.sleep(1./240.)
            
            # Now move to actual grasp position
            for step in range(50):
                ee_state = p.getLinkState(panda, 11)
                current_pos = ee_state[0]
                direction = np.array(target_pos) - np.array(current_pos)
                distance = np.linalg.norm(direction)
                
                if distance < 0.02:
                    break
                    
                joint_velocities = []
                for i in range(7):
                    if i < 3:
                        vel = direction[i] * 1.0
                    else:
                        vel = 0.0
                    joint_velocities.append(vel)
                
                p.setJointMotorControlArray(
                    panda, 
                    range(7), 
                    p.VELOCITY_CONTROL,
                    targetVelocities=joint_velocities,
                    forces=[100] * 7
                )
                
                p.stepSimulation()
                time.sleep(1./240.)
            
            # Close gripper to grasp
            print("Closing gripper...")
            p.setJointMotorControlArray(
                panda, 
                [9, 10], 
                p.POSITION_CONTROL,
                targetPositions=[0.0, 0.0],  # Close gripper
                forces=[100, 100]
            )
            
            # Let gripper close
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./240.)

        move_to_grasp(grasp[0])  # Take first prediction if batch
        
        # Keep simulation running for observation
        print("Simulation running. Close the window to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1./240.)
            
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Clean up temporary files
        for temp_file in ['temp_rgb.png', 'temp_depth.png']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        # Safely disconnect from physics server
        if p.isConnected():
            p.disconnect()

# --- Evaluation ---
def evaluate_model():
    rgb_paths = sorted(glob.glob('data/train_4/scene_0090/kinect/rgb/*.png'))[:10]
    depth_paths = sorted(glob.glob('data/train_4/scene_0090/kinect/depth/*.png'))[:10]
    meta_paths = sorted(glob.glob('data/train_4/scene_0090/kinect/meta/*.mat'))[:10]
    dataset = GraspDataset(rgb_paths, depth_paths, meta_paths)

    model = GraspViT()
    try:
        model.load_state_dict(torch.load('grasp_model.pth'))
    except FileNotFoundError:
        print("Error: grasp_model.pth not found. Please run training first.")
        return
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for rgbd, grasp_true in dataset:
        rgbd = rgbd.unsqueeze(0).to(device)
        with torch.no_grad():
            grasp_pred = model(rgbd).cpu().numpy()
        error = np.linalg.norm(grasp_pred[:3] - grasp_true[:3].numpy())
        print(f"Position error: {error:.4f}")

# --- Check .mat Keys ---
def check_mat_keys():
    meta_path = 'data/train_4/scene_0090/kinect/meta/0000.mat'
    try:
        meta = scipy.io.loadmat(meta_path)
        print("Meta file keys:", meta.keys())
    except FileNotFoundError:
        print(f"Error: {meta_path} not found.")

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp Prediction Pipeline")
    parser.add_argument('--task', choices=['explore', 'train', 'simulate', 'evaluate', 'check_keys'],
                        default='explore', help="Task to execute")
    args = parser.parse_args()

    if args.task == 'explore':
        explore_data()
    elif args.task == 'train':
        train_model()
    elif args.task == 'simulate':
        simulate_grasp()
    elif args.task == 'evaluate':
        evaluate_model()
    elif args.task == 'check_keys':
        check_mat_keys()