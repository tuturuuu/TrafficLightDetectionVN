import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')  # Use headless backend
import matplotlib.pyplot as plt

# --- Load trained model ---
model = YOLO("/home/vietpham/dataset/result/traffic_light_yolo3/weights/best.pt")

# --- Define dataset paths ---
dataset_path = "/home/vietpham/dataset/20240425-trafficlightandcountdowndisplay-1000/"
val_images = sorted(glob.glob(os.path.join(dataset_path, "val/images/*.jpg")))

# --- IoU helper function ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# --- Read YOLO txt labels ---
def read_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw/2) * img_w)
                y1 = int((y - bh/2) * img_h)
                x2 = int((x + bw/2) * img_w)
                y2 = int((y + bh/2) * img_h)
                boxes.append([cls, x1, y1, x2, y2])
    return boxes

# --- Draw YOLO boxes ---
def draw_yolo_boxes(image_path, label_path, color=(0, 255, 0), label_type="GT"):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
                x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{int(cls)}-{label_type}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img

# --- Evaluate & save wrong predictions ---
wrong_count = 0
max_save = 10  # save only 10 images

output_dir = os.path.join("~/wrong_predictions")
os.makedirs(output_dir, exist_ok=True)

for img_path in val_images:
    if wrong_count >= max_save:
        break

    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    gt_boxes = read_gt_boxes(label_path, w, h)
    results = model.predict(source=img_path, conf=0.25, verbose=False, iou=0.45)
    preds = results[0].boxes

    pred_boxes = []
    for b in preds:
        cls = int(b.cls[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        pred_boxes.append([cls, x1, y1, x2, y2])

    # --- Determine if prediction is wrong ---
    iou_threshold = 0.5
    wrong = False
    used_pred = set()

    # Check missing or wrong-class predictions
    for gt in gt_boxes:
        gt_cls, x1, y1, x2, y2 = gt
        matched = False
        for i, pred in enumerate(pred_boxes):
            pred_cls, px1, py1, px2, py2 = pred
            iou = compute_iou([x1, y1, x2, y2], [px1, py1, px2, py2])
            if iou >= iou_threshold:
                used_pred.add(i)
                if int(pred_cls) != int(gt_cls):
                    wrong = True  # wrong class
                matched = True
        if not matched:
            wrong = True  # missing detection

    # Check extra boxes
    if len(pred_boxes) > len(used_pred):
        wrong = True

    if not wrong:
        continue  # skip correct cases

    wrong_count += 1

    # --- Visualization ---
    gt_img = draw_yolo_boxes(img_path, label_path, color=(0, 255, 0), label_type="GT")
    pred_img = results[0].plot()  # predicted boxes in red

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Ground Truth (Green Boxes)")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Predicted (Red Boxes)")
    axs[1].axis("off")

    # Save figure to disk
    out_path = os.path.join(output_dir, f"wrong_{wrong_count}.png")
    plt.savefig(out_path)
    plt.close(fig)

print(f"Total wrong predictions saved: {wrong_count}")

