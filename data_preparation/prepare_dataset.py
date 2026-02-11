import argparse
from pathlib import Path
import random
import shutil
import yaml


def copy_files(file_list, images_root, dest_images, labels_root, dest_labels):
    for img in file_list:
        rel = img.relative_to(images_root)

        dst_img = dest_images / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)

        src_label = labels_root / rel.with_suffix('.txt')
        dst_label = dest_labels / rel.with_suffix('.txt')
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            open(dst_label, 'w').close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images_root', default='outputs/synth_images')
    p.add_argument('--labels_root', default='outputs/labels')
    p.add_argument('--out_dir', default='dataset')
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--val_dir', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_all_train', action='store_true')
    p.add_argument('--nc', type=int, default=1)
    p.add_argument('--names', nargs='*', default=['object'])
    args = p.parse_args()

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    out_dir = Path(args.out_dir)

    if not images_root.exists():
        raise SystemExit(f"Images root not found: {images_root}")

    all_images = sorted(images_root.rglob('*.png'))
    if not all_images:
        raise SystemExit(f"No images found in {images_root}")

    random.seed(args.seed)

    if args.val_dir and Path(args.val_dir).exists():
        train_imgs = all_images
        val_imgs = sorted(Path(args.val_dir).rglob('*.png'))
    elif args.use_all_train:
        train_imgs = all_images
        val_imgs = []
    else:
        random.shuffle(all_images)
        n_val = int(len(all_images) * args.val_ratio)
        val_imgs = all_images[:n_val]
        train_imgs = all_images[n_val:]

    copy_files(train_imgs, images_root, out_dir / 'images' / 'train', labels_root, out_dir / 'labels' / 'train')
    copy_files(val_imgs, images_root, out_dir / 'images' / 'val', labels_root, out_dir / 'labels' / 'val')

    data_yaml = {
        'train': str((out_dir / 'images' / 'train').resolve()),
        'val': str((out_dir / 'images' / 'val').resolve()),
        'nc': args.nc,
        'names': args.names
    }

    with open(out_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Train: {len(train_imgs)} images")
    print(f"Val:   {len(val_imgs)} images")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == '__main__':
    main()