import time
from ultralytics import YOLO
from pathlib import Path

model = YOLO("runs/traffic_light_dataset2_tiling/weights/best.pt")

image_dir = Path("/home/vietpham/dataset/20240425-trafficlightandcountdowndisplay-1000/test_tiled/images")
images = list(image_dir.glob("*.jpg"))

TILES_PER_IMAGE = 3  # !!!! change this to your actual tiling number

# ---- Warm-up ----
for _ in range(5):
    model.predict(images[0], batch=1, device=0, verbose=False)

# ---- Measure FULL tiled inference ----
start = time.perf_counter()

for img in images:
    # this represents ONE ORIGINAL IMAGE
    # internally you run inference on all its tiles
    for _ in range(TILES_PER_IMAGE):
        model.predict(img, batch=1, device=0, verbose=False)

end = time.perf_counter()

total_time = end - start

real_fps = len(images) / total_time
tile_fps = (len(images) * TILES_PER_IMAGE) / total_time

print(f"Original images processed: {len(images)}")
print(f"Tiles per image: {TILES_PER_IMAGE}")
print(f"Total time: {total_time:.3f} sec")

print(f"\nTile-level FPS: {tile_fps:.2f}")
print(f"✅ REAL image-level FPS (with tiling): {real_fps:.2f}")

# from ultralytics import YOLO

# model = YOLO("runs/traffic_light_dataset2_tiling/weights/best.pt")

# # Run evaluation
# metrics = model.val(
#     data="/home/vietpham/dataset/20240425-trafficlightandcountdowndisplay-1000/data_tiled.yaml",
#     imgsz=640,
#     batch=16
# )

# # Print evaluation results
# print("mAP50:", metrics.box.map50)
# print("mAP50-95:", metrics.box.map)
# print("Precision:", metrics.box.mp)
# print("Recall:", metrics.box.mr)
