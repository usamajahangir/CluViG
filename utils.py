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
    for epoch in range(10):
        for rgbd, grasp in dataloader:
            rgbd, grasp = rgbd.to(device), grasp.to(device)
            optimizer.zero_grad()
            outputs = model(rgbd)
            loss = criterion(outputs, grasp)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'grasp_model.pth')


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
