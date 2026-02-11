import cv2
import numpy as np
from pathlib import Path

rewrite = [77, 79,80,81,82,85,86,87,88,90,91,92]

for i in rewrite:
    out_dir = Path(f"outputs/bg_images/{i}")
    out_dir.mkdir(parents=True, exist_ok=True)  #чтоб не падал при наличии папки

    n = sum(p.is_file() for p in Path(f"images/{i}").iterdir())
    for j in range(1, n):
        img = cv2.imread(f"images/{i}/1({j}).png", cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(f"masks/{i}/1({j}).png", cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"skip: i={i}, j={j} (img or mask not found/readable)")
            continue

        mask255 = ((mask > 0).astype(np.uint8)) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dil = cv2.dilate(mask255, kernel, iterations=3) #расширение

        bg = cv2.inpaint(img, mask_dil, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        cv2.imwrite(str(out_dir / f"{j}.png"), bg)

###
#ffmpeg -framerate 15 -i %d.png -c:v libx264 -pix_fmt yuv420p "C:\Users\User\Desktop\real\real\outputs\bg_videos\4.mov"
###