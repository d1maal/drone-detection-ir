import cv2
import numpy as np
from pathlib import Path

def save_object_samples(video_id: int, frames: list[int],
                        out_dir="outputs/cut_object",
                        pad: int = 5,
                        save_mask: bool = True):
    out_dir = Path(out_dir) / str(video_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    for j in frames:
        img_path = Path(f"images/{video_id}/1({j}).png")
        mask_path = Path(f"masks/{video_id}/1({j}).png")

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"skip frame {j}: can't read img/mask")
            continue

        mask_bin = (mask > 0)
        ys, xs = np.where(mask_bin)
        if len(xs) == 0:
            print(f"skip frame {j}: empty mask")
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        H, W = img.shape
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)

        patch = img[y1:y2+1, x1:x2+1]

        cv2.imwrite(str(out_dir / f"obj_{j:04d}.png"), patch)

        if save_mask:
            patch_mask = (mask_bin[y1:y2+1, x1:x2+1].astype(np.uint8) * 255)
            cv2.imwrite(str(out_dir / f"obj_{j:04d}_mask.png"), patch_mask)

        print(f"saved frame {j}: bbox=({x1},{y1})-({x2},{y2}), size={patch.shape[1]}x{patch.shape[0]}")

save_object_samples(
    video_id=26,
    frames=[15, 32],   
    out_dir="outputs/cut_object",
    pad=5,
    save_mask=True
)
