import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import pybullet as p
import pybullet_data
import time

# Global variable to store the active grasp constraint
global_grasp_constraint_id = -1 

# --- Preprocessing ---
def preprocess_rgbd_from_arrays(rgb_array, depth_array):
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    print(f"Input RGB shape: {rgb_array.shape}")
    print(f"Input depth shape: {depth_array.shape}")
    
    rgb_resized = cv2.resize(rgb_array, (224, 224))
    depth_resized = cv2.resize(depth_array, (224, 224))
    
    if depth_resized.max() > depth_resized.min():
        depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255
    else:
        depth_normalized = np.zeros_like(depth_resized)
    depth_normalized = depth_normalized.astype(np.uint8)
    
    rgbd = rgb_resized.copy()
    rgbd[:, :, 0] = depth_normalized
    
    rgbd_pil = Image.fromarray(rgbd)
    inputs = processor(images=rgbd_pil, return_tensors="pt")
    return inputs['pixel_values']

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
    global global_grasp_constraint_id

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.setPhysicsEngineParameter(
        numSolverIterations=2000, 
        enableFileCaching=0
    )

    plane = p.loadURDF('plane.urdf')
    table = p.loadURDF('table/table.urdf', [0, 0, 0])
    
    panda = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.625], useFixedBase=True)
    
    num_joints = p.getNumJoints(panda)
    print(f"Robot has {num_joints} joints")
    
    initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for i in range(min(7, num_joints)):
        p.resetJointState(panda, i, initial_joint_positions[i])
    
    if num_joints > 7:
        p.resetJointState(panda, 9, 0.04)
        p.resetJointState(panda, 10, 0.04)
        p.changeDynamics(panda, 9, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)
        p.changeDynamics(panda, 10, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)

    objects = []
    fixed_position = [0.4, 0.0, 0.65]  # Fixed object position
    object_color = [1, 0, 0, 1]
    
    obj = p.loadURDF('cube_small.urdf', fixed_position, [0, 0, 0, 1])
    p.changeVisualShape(obj, -1, rgbaColor=object_color)
    p.changeDynamics(obj, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)
    objects.append(obj)
    print(f"Loaded object at fixed position {fixed_position}")
    
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)

    def get_end_effector_camera_view():
        ee_state = p.getLinkState(panda, 11)
        ee_pos = np.array(ee_state[0])
        ee_orn = ee_state[1]
        
        ee_rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        camera_offset_local = np.array([0.0, 0.0, 0.05])
        camera_pos = ee_pos + ee_rot_matrix @ camera_offset_local
        look_distance = 0.25
        look_direction_local = np.array([0.0, 0.0, look_distance])
        camera_target = camera_pos + ee_rot_matrix @ look_direction_local
        up_vector_local = np.array([0.0, 1.0, 0.0])
        camera_up = ee_rot_matrix @ up_vector_local
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=camera_up.tolist()
        )
        
        width, height = 224, 224
        fov = 60
        aspect = width / height
        near_plane = 0.01
        far_plane = 5.0
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near_plane,
            farVal=far_plane
        )
        
        images = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb = np.array(images[2]).reshape(height, width, 4)[:, :, :3]
        depth = np.array(images[3]).reshape(height, width)
        
        depth_real = far_plane * near_plane / (far_plane - (far_plane - near_plane) * depth)
        depth_real = np.clip(depth_real, 0, far_plane)
        
        return rgb.astype(np.uint8), depth_real

    model = GraspViT()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        print("Will use fixed position for grasping")
    else:
        model.to(device)
        model.eval()

    def move_robot_to_scan_position():
        scan_joint_positions = [0.0, -0.5, 0, -2.0, 0, 1.5, 0.785]
        for i in range(7):
            p.setJointMotorControl2(
                panda, i,
                p.POSITION_CONTROL,
                targetPosition=scan_joint_positions[i],
                force=500
            )
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1./240.)
        return scan_joint_positions

    def get_inverse_kinematics(target_pos, target_orn=None, rest_poses=None):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        joint_ranges = [ul - ll for ll, ul in zip(lower_limits, upper_limits)]
        
        if rest_poses is None:
            rest_poses = [0.0, -0.5, 0, -2.0, 0, 1.5, 0.785]
        
        joint_poses = p.calculateInverseKinematics(
            panda,
            11,
            target_pos,
            target_orn,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=2000,
            residualThreshold=0.0001
        )
        
        print(f"IK target position: {target_pos}, joints: {list(joint_poses[:7])}")
        return list(joint_poses)[:7]

    def move_to_position(target_pos, target_orn=None, rest_poses=None, max_steps=1500, distance_threshold=0.015):
        print(f"Moving to position: {target_pos}")
        target_joints = get_inverse_kinematics(target_pos, target_orn, rest_poses)
        
        for i in range(7):
            p.setJointMotorControl2(
                panda, i,
                p.POSITION_CONTROL,
                targetPosition=target_joints[i],
                force=1000,
                maxVelocity=2.0
            )
        
        for step in range(max_steps):
            current_pos = p.getLinkState(panda, 11)[0]
            current_joints = [p.getJointState(panda, i)[0] for i in range(7)]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            if distance < distance_threshold:
                print(f"Reached target position in {step} steps. Distance: {distance:.3f}m")
                return True
            if step % 100 == 0:
                print(f"Step {step}: Distance: {distance:.3f}m, Joints: {current_joints}")
            p.stepSimulation()
            time.sleep(1./240.)
        
        print(f"Failed to reach target. Distance to target: {distance:.3f}m")
        return False

    def execute_grasp(grasp_prediction=None):
        global global_grasp_constraint_id

        fixed_position = [0.4, 0.0, 0.65]
        workspace_center = np.array([0.4, 0.0, 0.65])
        workspace_scale = np.array([0.05, 0.05, 0.01])  # Tighter scale

        if grasp_prediction is not None and model_loaded:
            print(f"Using model prediction: {grasp_prediction}")
            target_pos = workspace_center + grasp_prediction[:3] * workspace_scale
            target_pos[2] = max(target_pos[2], 0.63) + 0.01  # Ensure above object
            max_deviation = 0.05
            if np.linalg.norm(target_pos - fixed_position) > max_deviation:
                print(f"Prediction too far from fixed position {fixed_position}, using fixed position")
                target_pos = fixed_position.copy()
                target_pos[2] += 0.01
                target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
            else:
                if len(grasp_prediction) >= 7:
                    target_orn = grasp_prediction[3:7]
                    target_orn = target_orn / np.linalg.norm(target_orn)
                else:
                    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        else:
            print("No model prediction or model not loaded, using fixed position...")
            target_pos = fixed_position.copy()
            target_pos[2] += 0.01
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        pre_grasp_pos = target_pos.copy()
        pre_grasp_pos[2] += 0.1
        print(f"Moving to pre-grasp position: {pre_grasp_pos}")
        if not move_to_position(pre_grasp_pos, target_orn, scan_joint_positions):
            print("Failed to reach pre-grasp position")
            return False

        print("Opening gripper...")
        p.setJointMotorControl2(panda, 9, p.POSITION_CONTROL, targetPosition=0.04, force=100)
        p.setJointMotorControl2(panda, 10, p.POSITION_CONTROL, targetPosition=0.04, force=100)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("Moving to grasp position...")
        if not move_to_position(target_pos, target_orn, scan_joint_positions):
            print("Failed to reach grasp position")
            return False
        
        print("Closing gripper...")
        p.setJointMotorControl2(panda, 9, p.POSITION_CONTROL, targetPosition=0.0, force=1000)
        p.setJointMotorControl2(panda, 10, p.POSITION_CONTROL, targetPosition=0.0, force=1000)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        object_to_grasp = objects[0] if objects else None
        if object_to_grasp is not None:
            contact_points_left = p.getContactPoints(panda, object_to_grasp, linkIndexA=9)
            contact_points_right = p.getContactPoints(panda, object_to_grasp, linkIndexA=10)
            if len(contact_points_left) + len(contact_points_right) > 0:
                print("Gripper made contact with object. Creating constraint...")
                try:
                    if global_grasp_constraint_id != -1:
                        p.removeConstraint(global_grasp_constraint_id)
                    global_grasp_constraint_id = p.createConstraint(
                        parentBodyUniqueId=panda,
                        parentLinkIndex=11,
                        childBodyUniqueId=object_to_grasp,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0]
                    )
                    print(f"Fixed constraint created with ID: {global_grasp_constraint_id}")
                except p.error as e:
                    print(f"Failed to create constraint: {e}")
                    return False
            else:
                print("No contact points detected. Grasp failed.")
                return False
        else:
            print("No object to grasp. Grasp failed.")
            return False

        print("Lifting object...")
        lift_pos = target_pos.copy()
        lift_pos[2] += 0.15
        if not move_to_position(lift_pos, target_orn, scan_joint_positions):
            print("Failed to lift object")
            return False
        
        print("Grasp execution completed!")
        return True

    try:
        print("Moving robot to initial scanning position...")
        scan_joint_positions = move_robot_to_scan_position()
        
        print("Capturing initial camera feed...")
        rgb, depth = get_end_effector_camera_view()
        
        cv2.imwrite('ee_camera_rgb.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite('ee_camera_depth.png', (depth / depth.max() * 255).astype(np.uint8))
        print("Saved camera images: ee_camera_rgb.png, ee_camera_depth.png")
        
        grasp_prediction = None
        if model_loaded:
            print("Processing image through grasp prediction model...")
            rgbd = preprocess_rgbd_from_arrays(rgb, depth)
            rgbd = rgbd.to(device)
            with torch.no_grad():
                grasp_prediction = model(rgbd).cpu().numpy()[0]
            print("Predicted grasp:", grasp_prediction)
        
        success = execute_grasp(grasp_prediction)
        
        if success:
            print("Grasp successful!")
        else:
            print("Grasp failed!")
        
        print("Simulation running. Press Ctrl+C or close window to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1./60.)
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if p.isConnected():
            if global_grasp_constraint_id != -1 and p.doesConstraintExist(global_grasp_constraint_id):
                p.removeConstraint(global_grasp_constraint_id)
                print(f"Removed constraint ID: {global_grasp_constraint_id}")
            p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp Prediction Pipeline")
    parser.add_argument('--task', choices=['simulate'], default='simulate', help="Task to execute")
    args = parser.parse_args()

    if args.task == 'simulate':
        simulate_grasp()