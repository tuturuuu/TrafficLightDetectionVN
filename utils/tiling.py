import os
import cv2
from pathlib import Path
from tqdm import tqdm

# ========== CONFIG ==========
dataset_path = "/home/vietpham/dataset/20240425-trafficlightandcountdowndisplay-1000"
splits = ["train", "val", "test"]    # process all 3 splits
tile_size = 640                      # tile dimension (640x640)
overlap = 0.2                        # overlap ratio between tiles
# ============================

def tile_image_and_labels(img_path, label_path, out_img_dir, out_lbl_dir, tile_size=640, overlap=0.2):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping unreadable image: {img_path}")
        return
    h, w = img.shape[:2]
    step = int(tile_size * (1 - overlap))
    base = Path(img_path).stem

    # read YOLO labels
    labels = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = map(float, parts)
                cx = x * w
                cy = y * h
                bw_px = bw * w
                bh_px = bh * h
                x1 = cx - bw_px / 2
                y1 = cy - bh_px / 2
                x2 = cx + bw_px / 2
                y2 = cy + bh_px / 2
                labels.append((int(cls), x1, y1, x2, y2))

    tile_id = 0
    for y in range(0, max(1, h - tile_size + 1), step):
        for x in range(0, max(1, w - tile_size + 1), step):
            x2 = min(w, x + tile_size)
            y2 = min(h, y + tile_size)
            x1 = x2 - tile_size
            y1 = y2 - tile_size
            tile = img[y1:y2, x1:x2]
            out_img = os.path.join(out_img_dir, f"{base}_tile{tile_id}.jpg")
            cv2.imwrite(out_img, tile)

            # remap labels to tile
            out_lbl = os.path.join(out_lbl_dir, f"{base}_tile{tile_id}.txt")
            with open(out_lbl, "w") as lf:
                for cls, bx1, by1, bx2, by2 in labels:
                    ix1 = max(bx1, x1)
                    iy1 = max(by1, y1)
                    ix2 = min(bx2, x2)
                    iy2 = min(by2, y2)
                    if ix2 > ix1 and iy2 > iy1:
                        cx = ((ix1 + ix2) / 2 - x1) / tile_size
                        cy = ((iy1 + iy2) / 2 - y1) / tile_size
                        bw_n = (ix2 - ix1) / tile_size
                        bh_n = (iy2 - iy1) / tile_size
                        if 0 < bw_n <= 1 and 0 < bh_n <= 1:
                            lf.write(f"{cls} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}\n")
            tile_id += 1


# ========== RUN TILING ==========
for split in splits:
    img_dir = os.path.join(dataset_path, f"{split}/images")
    lbl_dir = os.path.join(dataset_path, f"{split}/labels")
    out_img_dir = os.path.join(dataset_path, f"{split}_tiled/images")
    out_lbl_dir = os.path.join(dataset_path, f"{split}_tiled/labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"\n📸 Tiling {len(image_files)} {split} images...")

    for f in tqdm(image_files):
        img_path = os.path.join(img_dir, f)
        lbl_path = os.path.join(lbl_dir, Path(f).stem + ".txt")
        tile_image_and_labels(img_path, lbl_path, out_img_dir, out_lbl_dir, tile_size, overlap)

print("Done! Tiled dataset created.")
