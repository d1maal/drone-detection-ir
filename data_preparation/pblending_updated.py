import cv2
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

def spline_trajectory(keypoints: dict[int, tuple[int,int]], n_frames: int):
    f = np.array(sorted(keypoints), dtype=float)
    xs = np.array([keypoints[int(k)][0] for k in f], dtype=float)
    ys = np.array([keypoints[int(k)][1] for k in f], dtype=float)
    if f[0] != 1:  f = np.insert(f, 0, 1.0); xs = np.insert(xs, 0, xs[0]); ys = np.insert(ys, 0, ys[0]) #иначе упадёт
    if f[-1] != n_frames: f = np.append(f, float(n_frames)); xs = np.append(xs, xs[-1]); ys = np.append(ys, ys[-1])
    csx, csy = CubicSpline(f, xs, bc_type="natural"), CubicSpline(f, ys, bc_type="natural") #natural - более "натуральные" граничные условия на сплайне
    t = np.arange(1, n_frames + 1, dtype=float)
    return [(int(round(x)), int(round(y))) for x, y in zip(csx(t), csy(t))]

def clamp_center(cx, cy, obj_w, obj_h, W, H):
    hw, hh = obj_w // 2, obj_h // 2
    return (max(hw + 1, min(W - hw - 2, cx)), max(hh + 1, min(H - hh - 2, cy))) 

def poisson_insert(bg_gray, obj_gray, obj_mask, center_xy, mode="mixed", mask_dilate=1):
    bg = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR) #seamlessClone требует цветные изображения
    obj = cv2.cvtColor(obj_gray, cv2.COLOR_GRAY2BGR)
    mask = ((obj_mask > 0).astype(np.uint8) * 255)
    if mask_dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        mask = cv2.dilate(mask, k, iterations=int(mask_dilate))
    flag = cv2.MIXED_CLONE if mode == "mixed" else cv2.NORMAL_CLONE #mixed сохр градиенты объекта (как раз нужно для вставки Пуассоном)
    out = cv2.seamlessClone(obj, bg, mask, (int(center_xy[0]), int(center_xy[1])), flag)
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

def place_full_mask(H, W, obj_mask, cx, cy):
    h, w = obj_mask.shape
    x1, y1 = int(cx - w // 2), int(cy - h // 2)
    x2, y2 = x1 + w, y1 + h
    full = np.zeros((H, W), dtype=np.uint8)

    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(W, x2), min(H, y2)
    ox1, oy1 = fx1 - x1, fy1 - y1
    ox2, oy2 = ox1 + (fx2 - fx1), oy1 + (fy2 - fy1)

    if fx1 < fx2 and fy1 < fy2:
        full[fy1:fy2, fx1:fx2] = ((obj_mask[oy1:oy2, ox1:ox2] > 0).astype(np.uint8) * 255)
    return full

def write_txt(path: Path, nums):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(map(str, nums)) + "\n", encoding="utf-8")


video_id = 90
bg_dir = Path(f"outputs/bg_images/{video_id}")
out_img = Path(f"outputs/synth_images/{video_id}")
out_msk = Path(f"outputs/synth_masks/{video_id}")
out_box = Path(f"outputs/synth_boxes/{video_id}")
out_ctr = Path(f"outputs/synth_center/{video_id}")
for d in (out_img, out_msk, out_box, out_ctr):
    d.mkdir(parents=True, exist_ok=True)

obj = cv2.imread("outputs/cut_object/55/obj_0022.png", cv2.IMREAD_GRAYSCALE)
obj_mask = cv2.imread("outputs/cut_object/55/obj_0022_mask.png", cv2.IMREAD_GRAYSCALE)
if obj is None or obj_mask is None:
    raise FileNotFoundError("Object patch or mask not found")
obj_h, obj_w = obj.shape

n_frames = len(list(bg_dir.glob("*.png")))
if n_frames == 0:
    raise ValueError(f"No background frames found in {bg_dir}")

keypoints = {
    1: (61, 176),
    n_frames//4: (205, 164),
    n_frames//2: (355, 171),
    2*n_frames//3: (340, 133),
    3*n_frames//4: (537, 204),
    n_frames: (626, 180)
}
traj = spline_trajectory(keypoints, n_frames)
count_frames = 1

for j in range(1, n_frames + 1):
    bg = cv2.imread(str(bg_dir / f"{j}.png"), cv2.IMREAD_GRAYSCALE)
    if bg is None:
        continue

    H, W = bg.shape
    cx, cy = clamp_center(*traj[j - 1], obj_w, obj_h, W, H)

    obj_use = cv2.convertScaleAbs(obj, alpha=1.2, beta=0) #контраст

    res = poisson_insert(bg, obj_use, obj_mask, (cx, cy), mode="mixed", mask_dilate=1)
    cv2.imwrite(str(out_img / f"{count_frames}.png"), res)

    full_mask = place_full_mask(H, W, obj_mask, cx, cy)
    cv2.imwrite(str(out_msk / f"{count_frames}.png"), full_mask)

    write_txt(out_ctr / f"{count_frames}.txt", (cx, cy))

    ys, xs = np.where(full_mask > 0)
    if len(xs) == 0:
        write_txt(out_box / f"{count_frames}.txt", (0, 0, 0, 0))
    else:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        write_txt(out_box / f"{count_frames}.txt", (x1, y1, x2 - x1 + 1, y2 - y1 + 1))

    count_frames += 1
print(f"Done: {out_img} (+ masks/boxes/center)")