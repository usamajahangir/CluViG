import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

rgb_paths = sorted(glob.glob('data/train_4/*/kinect/rgb/*.png'))
depth_paths = sorted(glob.glob('data/train_4/*/kinect/depth/*.png'))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.canvas.mpl_connect('key_press_event', lambda event: [fig.canvas.manager.window.close() if event.key == 'q' else None])

def show_image(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    ax[0].imshow(rgb)
    ax[0].set_title('RGB Image')
    ax[1].imshow(depth, cmap='gray')
    ax[1].set_title('Depth Map')
    plt.draw()

index = 0

def on_press(event):
    global index
    if event.key == 'left':
        index = (index - 1) % len(rgb_paths)
    elif event.key == 'right':
        index = (index + 1) % len(rgb_paths)
    show_image(rgb_paths[index], depth_paths[index])

fig.canvas.mpl_connect('key_press_event', on_press)

show_image(rgb_paths[index], depth_paths[index])
plt.show()
