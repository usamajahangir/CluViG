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
def preprocess_rgbd(rgb, depth):
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Ensure RGB is in correct format
    if len(rgb.shape) == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    
    # Resize both RGB and depth to 224x224
    rgb_resized = cv2.resize(rgb, (224, 224))
    depth_resized = cv2.resize(depth, (224, 224))
    
    # Normalize depth to [0, 255]
    depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-6) * 255
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # Blend depth into RGB (use depth as red channel, keep green/blue)
    rgbd = rgb_resized.copy()
    rgbd[:, :, 0] = depth_normalized
    
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
        rgbd = preprocess_rgbd(cv2.imread(self.rgb_paths[idx]), cv2.imread(self.depth_paths[idx], cv2.IMREAD_GRAYSCALE))
        meta = scipy.io.loadmat(self.meta_paths[idx])
        grasps = meta.get('grasps', np.zeros((1, 7)))
        grasp = grasps[np.argmax(grasps[:, -1])] if grasps.shape[0] > 0 else grasps[0]
        grasp_tensor = torch.tensor(grasp[:7], dtype=torch.float32)
        return rgbd.squeeze(0), grasp_tensor

# --- Model ---
class GraspViT(nn.Module):
    def __init__(self):
        super(GraspViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(768, 7)

    def forward(self, x):
        outputs = self.vit(x).last_hidden_state[:, 0, :]
        grasp = self.fc(outputs)
        return grasp

# --- Simulation ---
def simulate_grasp():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load environment
    plane = p.loadURDF('plane.urdf')
    table = p.loadURDF('table/table.urdf', [0, 0, 0])
    panda = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.625], useFixedBase=True)
    
    # Get robot info
    num_joints = p.getNumJoints(panda)
    print(f"Robot has {num_joints} joints")
    
    # Set initial pose
    initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for i in range(min(7, num_joints)):
        p.resetJointState(panda, i, initial_joint_positions[i])
    
    # Close gripper initially
    if num_joints > 7:
        p.resetJointState(panda, 9, 0.04)
        p.resetJointState(panda, 10, 0.04)

    # Add objects
    objects = []
    for i in range(3):
        obj = p.loadURDF('cube_small.urdf', [0.4 + i*0.15, 0.1, 0.7], [0, 0, 0, 1])
        objects.append(obj)
    
    # Let physics settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)

    def get_end_effector_rgbd():
        width, height = 224, 224
        # Get end effector state
        ee_state = p.getLinkState(panda, 11)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]
        
        # Position camera slightly forward from end effector
        camera_offset = np.array([0, 0, -0.1])  # 10cm in front of end effector
        camera_pos = np.array(ee_pos) + np.array(p.multiplyTransforms([0, 0, 0], ee_orn, camera_offset, [0, 0, 0, 1])[0])
        
        # Camera looks towards workspace (down)
        target_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.5]
        view_matrix = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=2.0)
        
        images = p.getCameraImage(width, height, view_matrix, proj_matrix)
        rgb = np.array(images[2])
        depth = np.array(images[3])
        
        # Handle RGB data
        if len(rgb.shape) == 1:
            if rgb.size == width * height * 4:
                rgb = rgb.reshape(height, width, 4)[:, :, :3]
            elif rgb.size == width * height * 3:
                rgb = rgb.reshape(height, width, 3)
            else:
                raise ValueError(f"Unexpected RGB size: {rgb.size}")
        elif len(rgb.shape) == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        
        # Handle depth data
        if len(depth.shape) == 1:
            depth = depth.reshape(height, width)
        
        rgb = rgb.astype(np.uint8)
        depth = (depth * 255).astype(np.uint8)
        
        return rgb, depth

    model = GraspViT()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
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
        # Move to initial scanning position
        initial_pos = [0.5, 0, 0.9]
        for step in range(100):
            ee_state = p.getLinkState(panda, 11)
            current_pos = ee_state[0]
            direction = np.array(initial_pos) - np.array(current_pos)
            if np.linalg.norm(direction) < 0.05:
                break
            joint_velocities = [direction[i] * 2.0 if i < 3 else 0.0 for i in range(7)]
            p.setJointMotorControlArray(
                panda, range(7), p.VELOCITY_CONTROL,
                targetVelocities=joint_velocities, forces=[100] * 7)
            p.stepSimulation()
            time.sleep(1./240.)

        rgb, depth = get_end_effector_rgbd()
        cv2.imwrite('temp_rgb.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite('temp_depth.png', depth)
        rgbd = preprocess_rgbd(rgb, depth)
        rgbd = rgbd.to(device)
        
        with torch.no_grad():
            grasp = model(rgbd).cpu().numpy()
        print("Predicted grasp:", grasp)

        def move_to_grasp(grasp_pred):
            position = grasp_pred[:3]
            quaternion = grasp_pred[3:7]
            
            target_pos = [
                0.4 + position[0] * 0.3,
                position[1] * 0.3,
                0.8 + position[2] * 0.2
            ]
            
            # Move to pre-grasp position
            pre_grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.1]
            for step in range(100):
                ee_state = p.getLinkState(panda, 11)
                current_pos = ee_state[0]
                direction = np.array(pre_grasp_pos) - np.array(current_pos)
                if np.linalg.norm(direction) < 0.05:
                    break
                joint_velocities = [direction[i] * 2.0 if i < 3 else np.random.uniform(-0.1, 0.1) for i in range(7)]
                p.setJointMotorControlArray(
                    panda, range(7), p.VELOCITY_CONTROL,
                    targetVelocities=joint_velocities, forces=[100] * 7)
                p.stepSimulation()
                time.sleep(1./240.)
            
            # Move to grasp position
            for step in range(50):
                ee_state = p.getLinkState(panda, 11)
                current_pos = ee_state[0]
                direction = np.array(target_pos) - np.array(current_pos)
                if np.linalg.norm(direction) < 0.02:
                    break
                joint_velocities = [direction[i] * 1.0 if i < 3 else 0.0 for i in range(7)]
                p.setJointMotorControlArray(
                    panda, range(7), p.VELOCITY_CONTROL,
                    targetVelocities=joint_velocities, forces=[100] * 7)
                p.stepSimulation()
                time.sleep(1./240.)
            
            # Close gripper
            print("Closing gripper...")
            p.setJointMotorControlArray(
                panda, [9, 10], p.POSITION_CONTROL,
                targetPositions=[0.0, 0.0], forces=[100, 100])
            
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./240.)

        move_to_grasp(grasp[0])
        
        print("Simulation running. Close the window to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1./240.)
            
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        for temp_file in ['temp_rgb.png', 'temp_depth.png']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if p.isConnected():
            p.disconnect()

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp Prediction Pipeline")
    parser.add_argument('--task', choices=['explore', 'train', 'simulate', 'evaluate', 'check_keys'],
                        default='explore', help="Task to execute")
    args = parser.parse_args()

    if args.task == 'simulate':
        simulate_grasp()