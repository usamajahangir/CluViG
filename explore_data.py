import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import xml.etree.ElementTree as ET
import scipy.io

# Load file paths
rgb_paths = sorted(glob.glob('data/train_4/*/kinect/rgb/*.png'))
depth_paths = sorted(glob.glob('data/train_4/*/kinect/depth/*.png'))
meta_paths = sorted(glob.glob('data/train_4/*/kinect/meta/*.mat'))  # Assuming grasps in .mat
annotations_paths = sorted(glob.glob('data/train_4/*/kinect/annotations/*.xml'))

# Verify paths match
assert len(rgb_paths) == len(depth_paths) == len(meta_paths) == len(annotations_paths), \
    "Mismatch in number of files across datasets"

# Set up plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'q' else None)

def show_image(rgb_path, depth_path, meta_path, annotation_path):
    # Clear previous plots
    ax[0].clear()
    ax[1].clear()

    # Load RGB and depth images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    # Load grasp from .mat file (adjust 'grasps' key based on .mat structure)
    meta = scipy.io.loadmat(meta_path)
    grasps = meta.get('grasps', np.zeros((1, 7)))  # Fallback if key not found
    grasp = grasps[np.argmax(grasps[:, -1])] if grasps.shape[0] > 0 else grasps[0]  # Highest score
    grasp = grasp[:7]  # x, y, z, quaternion

    # Parse XML annotations
    annotation = ET.parse(annotation_path).getroot()
    objects = []
    for obj in annotation.findall('obj'):
        obj_id = obj.find('obj_id').text
        obj_name = obj.find('obj_name').text
        pos = np.array(obj.find('pos_in_world').text.split(), dtype=float)
        ori = np.array(obj.find('ori_in_world').text.split(), dtype=float)
        objects.append({'id': obj_id, 'name': obj_name, 'pos': pos, 'ori': ori})

    # Display images
    ax[0].imshow(rgb)
    ax[0].set_title(f'RGB Image\nGrasp: {grasp[:3]}')
    ax[1].imshow(depth, cmap='gray')
    ax[1].set_title('Depth Map')

    # Print object annotations
    print(f"\nImage: {rgb_path}")
    print("Objects:")
    for obj in objects:
        print(f"  ID: {obj['id']}, Name: {obj['name']}, Pos: {obj['pos']}, Ori: {obj['ori']}")
    print(f"Grasp (x, y, z, quaternion): {grasp}")

    plt.draw()

# Initial image
index = 0
def on_press(event):
    global index
    if event.key == 'left':
        index = (index - 1) % len(rgb_paths)
    elif event.key == 'right':
        index = (index + 1) % len(rgb_paths)
    show_image(rgb_paths[index], depth_paths[index], meta_paths[index], annotations_paths[index])

fig.canvas.mpl_connect('key_press_event', on_press)

# Show first image
show_image(rgb_paths[index], depth_paths[index], meta_paths[index], annotations_paths[index])
plt.show()