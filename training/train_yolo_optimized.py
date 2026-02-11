#!/usr/bin/env python3
"""
Optimized YOLOv8 training script for SMALL OBJECT DETECTION (drones in IR).

Changes from basic script:
- Larger image size (1024 instead of 640)
- Disabled mosaic augmentation (bad for small objects)
- Lower learning rate (more stable)
- Special parameters for small object detection
- More detailed logging

Usage:
  python train_yolo_optimized.py --epochs 20
  python train_yolo_optimized.py --epochs 100 --model yolov8m.pt
"""
import argparse
from pathlib import Path
import subprocess

def train_api(data_yaml, epochs, imgsz, model, batch, patience):
    """Train using Python API with optimized parameters for small objects."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('ultralytics import failed:', e)
        print('Install: pip install ultralytics')
        return False
    
    print("=" * 60)
    print("DRONE DETECTION TRAINING - OPTIMIZED FOR SMALL OBJECTS")
    print("=" * 60)
    print(f"Mdel: {model}")
    print(f"Image size: {imgsz}px (larger = better for small objects)")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Patience: {patience} (early stopping)")
    print("=" * 60)
    
    # Load model
    y = YOLO(model)
    
    # Train with optimized parameters for small object detection
    results = y.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        
        # PROJECT STRUCTURE
        project=str(Path(data_yaml).parent / 'runs'),
        name='exp',
        exist_ok=False,  # Create new folder for each run

        workers=0,                    # NO multiprocessing (fixes WinError 1455)
        
        # LEARNING RATE (lower = more stable for difficult tasks)
        lr0=0.001,      # Initial learning rate (default: 0.01)
        lrf=0.01,       # Final learning rate fraction
        
        # AUGMENTATIONS - OPTIMIZED FOR SMALL OBJECTS
        mosaic=0.0,     # DISABLED! Bad for small objects (default: 1.0)
        mixup=0.0,      # Disabled (can confuse small objects)
        copy_paste=0.0, # Disabled
        
        scale=0.2,
        translate=0.05,
        degrees=5.0,
        
        flipud=0.5,     # Vertical flip (good for drones in sky)
        fliplr=0.5,     # Horizontal flip
        
        # HSV augmentations (minimal for infrared)
        hsv_h=0.0,      # Hue shift (not needed for IR)
        hsv_s=0.0,      # Saturation (not needed for IR)
        hsv_v=0.2,      # Value/brightness variation (moderate for IR)
        
        # OPTIMIZER
        optimizer='AdamW',  # Better than SGD for difficult tasks
        cos_lr=True,        # Cosine LR scheduler
        
        # WARMUP (stabilize training start)
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        
        # LOSS WEIGHTS (emphasize box localization)
        box=7.5,        # Box loss weight (default: 7.5)
        cls=0.5,        # Classification loss weight (default: 0.5)
        dfl=1.5,        # DFL loss weight (default: 1.5)
        
        # REGULARIZATION
        weight_decay=0.001,
        dropout=0.4,           
        label_smoothing=0.05,  
        
        # TRAINING FEATURES
        amp=False,      # Disable mixed precision (more stable)
        save=True,
        save_period=-1, # Save only best and last
        plots=True,     # Generate plots
        verbose=True,
        
        # VALIDATION
        val=True,
        
        # MULTI-GPU (if available)
        device='cuda',    # Auto-select
    )
    
    print("=" * 60)
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    print("=" * 60)
    print("\nFinal metrics:")
    print(f"  mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    print("=" * 60)
    
    return True


def train_cli(data_yaml, epochs, imgsz, model, batch, patience):
    """Fallback: train using CLI with optimized parameters."""
    cmd = [
        'yolo', 'detect', 'train',
        f'data={str(data_yaml)}',
        f'model={model}',
        f'epochs={epochs}',
        f'imgsz={imgsz}',
        f'batch={batch}',
        f'patience={patience}',
        f'project={str(Path(data_yaml).parent/"runs")}',
        'name=exp',
        
        # Optimizations
        'lr0=0.001',
        'mosaic=0.0',
        'scale=0.2',
        'translate=0.05',
        'flipud=0.5',
        'degrees=10.0',
        'hsv_h=0.0',
        'hsv_s=0.0',
        'hsv_v=0.2',
        'optimizer=AdamW',
        'cos_lr=True',
        'amp=False',
    ]
    
    print('Running CLI fallback command:')
    print(' '.join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print('CLI train failed:', e)
        return False


def main():
    p = argparse.ArgumentParser(
        description='Train YOLOv8 for drone detection (optimized for small objects)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (20 epochs)
  python train_yolo_optimized.py --epochs 20
  
  # Full training with larger model
  python train_yolo_optimized.py --epochs 100 --model yolov8m.pt
  
  # Maximum quality (slow, needs powerful GPU)
  python train_yolo_optimized.py --epochs 100 --model yolov8l.pt --imgsz 1280
        """
    )
    
    # Required
    p.add_argument('--data_yaml', 
                   default='dataset/data.yaml',
                   help='Path to data.yaml config file')
    
    # Model architecture
    p.add_argument('--model', 
                   default='yolo26s.pt',
                   choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                   help='Model size (s=small recommended for start)')
    
    # Training parameters
    p.add_argument('--epochs', 
                   type=int, 
                   default=20,
                   help='Number of training epochs')
    
    p.add_argument('--imgsz', 
                   type=int, 
                   default=1024,
                   help='Input image size (1024 or 1280 for small objects)')
    
    p.add_argument('--batch', 
                   type=int, 
                   default=4,
                   help='Batch size (reduce if GPU out of memory)')
    
    p.add_argument('--patience',
                   type=int,
                   default=10,
                   help='Early stopping patience (stop if no improvement)')
    
    args = p.parse_args()
    
    # Validate data.yaml exists
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f'data.yaml not found: {data_yaml}')
    
    # Print configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Data config: {data_yaml}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}px")
    print(f"Batch size: {args.batch}")
    print(f"Patience: {args.patience}")
    print("=" * 60)
    
    # Try API first, fallback to CLI
    ok = train_api(data_yaml, args.epochs, args.imgsz, args.model, args.batch, args.patience)
    if not ok:
        print("\nAPI training failed, trying CLI fallback...\n")
        train_cli(data_yaml, args.epochs, args.imgsz, args.model, args.batch, args.patience)


if __name__ == '__main__':
    main()
