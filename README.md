# Drone Detection in Infrared Video

Detection of drones in infrared video using YOLO trained on synthetic data.

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | 0.496 |
| Precision | 0.636 |
| Recall | 0.507 |

##Images
| Исходное изображение | Удаление | Вставка(Poisson Blending) |
|:---:|:---:|:---:|
| ![Original](<img width="992" height="742" alt="Image" src="https://github.com/user-attachments/assets/564f225f-977e-4829-82bb-cca33b483822" />) | ![Delete](<img width="992" height="742" alt="Image" src="https://github.com/user-attachments/assets/3566599f-655e-4504-85be-ec725f9a19db" />) | [Poisson](<img width="992" height="742" alt="Image" src="https://github.com/user-attachments/assets/43ec579f-b119-4dba-a922-cb8c6623ff6f" />)  |

## Pipeline
```
Real IR video
     ↓
inpaint.py            Remove drones → clean background
cut_object.py         Extract drone patches with masks
pblending_updated.py  Insert drones along trajectory (Poisson blending)
generate_yolo_labels.py  Convert bbox to YOLO format
prepare_dataset.py    Split into train/val, create data.yaml
train_yolo_optimized.py  Train YOLO model
     ↓
best.pt
```

## Stack

- Python 3.11.9
- OpenCV
- Ultralytics YOLO
- PyTorch
- scipy

## Installation
```bash
pip install ultralytics opencv-python scipy pyyaml
```

## Usage

### 1. Prepare background (remove real drones)
```bash
python data_preparation/inpaint.py
```

### 2. Extract drone patches
```bash
python data_preparation/cut_object.py
```

### 3. Synthesize new frames
```bash
python data_preparation/pblending_updated.py
```

### 4. Generate YOLO labels
```bash
python data_preparation/generate_yolo_labels.py
```

### 5. Prepare dataset structure
```bash
python data_preparation/prepare_dataset.py --val_ratio 0.2
```

### 6. Train
```bash
python training/train_yolo_optimized.py --epochs 50 --imgsz 1024
```

## Dataset

- Train: 12673 images
- Val: 2800 images
- Class: drone (1 class)
- Image type: infrared (grayscale)
