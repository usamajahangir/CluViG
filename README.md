# CluViG
Vision-Based Grasping with a Transformer in a Cluttered Environment

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

#### Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```
#### Windows (CMD):
```bash
.\venv\Scripts\activate.bat
```
#### macOS/Linux:
```bash
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
ℹ️ For CUDA support (e.g., CUDA 11.8), ensure you have compatible drivers and install PyTorch with:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Run the Project