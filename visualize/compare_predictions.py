import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np


def parse_yolo_label(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cx, cy, w, h = map(float, parts[1:5])
            
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            
            boxes.append((x1, y1, x2, y2))
    
    return boxes


def compare_gt_pred(model_path, images_dir, labels_dir, output_dir, conf_threshold=0.25, num_samples=20):
    model = YOLO(model_path)
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(images_dir.rglob('*.png'))
    
    step = max(1, len(images) // num_samples)
    sampled = images[::step][:num_samples]
    
    for i, img_path in enumerate(sampled, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        rel_path = img_path.relative_to(images_dir)
        label_path = labels_dir / rel_path.with_suffix('.txt')
        
        gt_boxes = parse_yolo_label(label_path, w, h)
        
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        pred_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            pred_boxes.append((x1, y1, x2, y2, conf))
        
        gt_img = img.copy()
        for x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(gt_img, 'GT', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        pred_img = img.copy()
        for x1, y1, x2, y2, conf in pred_boxes:
            cv2.rectangle(pred_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(pred_img, f'{conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        combined = np.hstack([gt_img, pred_img])
        
        label_y = 30
        cv2.putText(combined, 'Ground Truth', (20, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(combined, 'Prediction', (w + 20, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        output_path = output_dir / f'compare_{i:04d}.png'
        cv2.imwrite(str(output_path), combined)
        
        print(f'Processed {i}/{len(sampled)}: GT={len(gt_boxes)} boxes, Pred={len(pred_boxes)} boxes')
    
    print(f'\nDone! Comparisons saved to {output_dir}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='runs/detect/exp/weights/best.pt')
    p.add_argument('--images', default='dataset/images/val')
    p.add_argument('--labels', default='dataset/labels/val')
    p.add_argument('--output', default='visualizations/comparisons')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--samples', type=int, default=20)
    args = p.parse_args()
    
    compare_gt_pred(args.model, args.images, args.labels, args.output, args.conf, args.samples)


if __name__ == '__main__':
    main()
