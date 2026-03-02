import ultralytics
from ultralytics import YOLO
import os
import cv2
import random
import matplotlib.pyplot as plt

# dataset path
dataset_path = "/home/vietpham/dataset/20240425-trafficlightandcountdowndisplay-1000"
image_dir = os.path.join(dataset_path, "train_tiled/images/")
label_dir = os.path.join(dataset_path, "train_tiled/labels")



# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
print(f"Found {len(image_files)} training images.")

class_names = [
    "tlr", "tly", "tlg", "tlra", "tlya", "tlga", "tsra", "tsya", "tsga",
    "trra", "trya", "trga", "tura", "tuya", "tuga", "tln", "tlr", "tly", "tlg",
    "tlra", "tlya", "tlga", "tsra", "tsya", "tsga", "trra", "trya", "trga",
    "tura", "tuya", "tuga", "tln",
    "117", "118", "119", "120", "121", "122", "123", "124", "125", "126",
    "127", "128", "129", "130", "131", "132", "133", "134", "135", "136",
    "137", "138", "139", "140", "141",
    "tnr", "tny", "tng", "tcr", "tcg", "tcn",
    "tllv", "tllvr", "tllvy", "tllvg", "tlrv", "tlrvr", "tlrvy", "tlrvg",
    "tlbv", "tclv", "tclvr", "tclvg", "tcrv", "tcrvr", "tcrvg", "tcbv"
]

yaml_path = os.path.join(dataset_path, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""
path: {dataset_path}
train: train_tiled/images
val: val_tiled/images
test: test_tiled/images

names:
""")
    for i, name in enumerate(class_names):
        f.write(f"  {i}: {name}\n")

print(open(yaml_path).read())

# Choose model size: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large)
model = YOLO('yolov8m.pt')  # start from pretrained COCO weights
# model = YOLO('./runs/traffic_light_yolo_tiling_bosch3/weights/last.pt')



# Check Ultralytics version
print("Ultralytics package version:", ultralytics.__version__)

# Print model details to see what version/core it's using
print("Model type:", type(model))
print("Model info:", model.info())

try:
    print("Model YAML path:", model.model.yaml_file)
except Exception as e:
    print("Could not get YAML file:", e)

# Train model
model.train(
    data=yaml_path,
    epochs=400,             # increase if have GPU time
    imgsz=640,             # image size (can change to 416 or 1280)
    batch=16,
    name='traffic_light_dataset2_tiling',
    project='./runs',  # where to save results
    device=0,# use GPU if available
    verbose=True,
)

# Evaluate on validation set
metrics = model.val()
print(metrics)

# Visualize predictions
results = model.predict(source=f"{dataset_path}/test_tiled/images", conf=0.25, save=True)
print(metrics.results_dict)
