
import argparse
from pathlib import Path
import cv2

def parse_box_line_xyhw_from_line(line): #x y h w 
    import re
    nums = re.findall(r"\d+\.?\d*", line)
    if len(nums) < 4:
        raise ValueError('not enough numeric values')
    x = float(nums[0])
    y = float(nums[1])
    h = float(nums[2])
    w = float(nums[3])
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return x_min, y_min, x_max, y_max


def to_yolo_line(x_min, y_min, x_max, y_max, img_w, img_h, class_id=0):
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images_root', default='outputs/synth_images', help='root folder with synthesized images')
    p.add_argument('--boxes_root', default='outputs/synth_boxes', help='root folder with synth box txt files')
    p.add_argument('--out_labels_root', default='outputs/labels', help='where to write yolo labels')
    p.add_argument('--class_id', type=int, default=0)
    args = p.parse_args()

    images_root = Path(args.images_root)
    boxes_root = Path(args.boxes_root)
    out_root = Path(args.out_labels_root)

    if not images_root.exists(): # bool
        raise SystemExit(f"Images root not found: {images_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(images_root.rglob('*.png')):
        rel = img_path.relative_to(images_root) # относительный путь 
        box_file = boxes_root / rel.with_suffix('.txt')
        label_file = out_root / rel.with_suffix('.txt')
        label_file.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: can't read image {img_path}")
            open(label_file, 'w').close()
            continue
        h_img, w_img = img.shape[:2]

        if not box_file.exists():
            open(label_file, 'w').close()
            continue

        lines = []
        with open(box_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    x_min, y_min, x_max, y_max = parse_box_line_xyhw_from_line(line)
                except Exception as e:
                    print(f"skip bad line in {box_file}: {line} -> {e}")
                    continue
                yolo = to_yolo_line(x_min, y_min, x_max, y_max, w_img, h_img, args.class_id)
                lines.append(yolo)

        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))

    print('YOLO labels written to', out_root)


if __name__ == '__main__':
    main()
