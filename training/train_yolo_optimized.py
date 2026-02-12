import argparse
from pathlib import Path

def train_api(data_yaml, epochs, imgsz, model, batch, patience):
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
    
    
    y = YOLO(model)
    
   
    results = y.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        
       
        project=str(Path(data_yaml).parent / 'runs'),
        name='exp',
        exist_ok=False, 

        workers=0,                    #multiprocessing 
        
       
        lr0=0.001,      
        lrf=0.01,       
        
        mosaic=0.0,     
        mixup=0.0,     
        copy_paste=0.0, 
        
        scale=0.2,
        translate=0.05,
        degrees=5.0,
        
        flipud=0.5,     
        fliplr=0.5,     
        
        
        hsv_h=0.0,     
        hsv_s=0.0,     
        hsv_v=0.2,      
        
        # OPTIMIZER
        optimizer='AdamW',  
        cos_lr=True,       
        
       
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        
        
        box=7.5,        
        cls=0.5,        
        dfl=1.5,        
        
        # REGULARIZATION
        weight_decay=0.001,
        dropout=0.4,           
        label_smoothing=0.05,  
        
       
        amp=False,     
        save=True,
        save_period=-1, 
        plots=True,    
        verbose=True,
        
        val=True,
        
       
        device='cuda',    
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

def main():
    p = argparse.ArgumentParser(
        description='Train YOLOv8 for drone detection (optimized for small objects)',
    )
    
    p.add_argument('--data_yaml', 
                   default='dataset/data.yaml',
                   help='Path to data.yaml config file')
    
    p.add_argument('--model', 
                   default='yolo26s.pt',
                   choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                   help='Model size (s=small recommended for start)')
    
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
    
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f'data.yaml not found: {data_yaml}')
    
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
    
    train_api(data_yaml, args.epochs, args.imgsz, args.model, args.batch, args.patience)


if __name__ == '__main__':
    main()
