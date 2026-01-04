from pathlib import Path
import os
import yaml
import random
import shutil
import cv2

# Update these paths according to your setup
BSTLD_ROOT = "/home/vietpham/dataset/bstld"  # Update this!
OUTPUT_ROOT = "./bstld_yolo_format"  # Update this!
RESULTS_DIR = "./bosch_result/results"  # Update this!

# BSTLD class mapping
# BSTLD has 4 main classes with various directional variants
CLASS_NAMES = ['red', 'yellow', 'green', 'off']

# Map all BSTLD label variants to the 4 main classes
CLASS_MAPPING = {
    # Lowercase versions
    'red': 0,
    'yellow': 1, 
    'green': 2,
    'off': 3,
    # Capitalized versions (actual BSTLD format)
    'Red': 0,
    'Yellow': 1,
    'Green': 2,
    'off': 3,  # 'off' is typically lowercase in BSTLD
    # Red variants
    'RedLeft': 0,
    'RedRight': 0,
    'RedStraight': 0,
    'RedStraightLeft': 0,
    'RedStraightRight': 0,
    # Yellow variants
    'YellowLeft': 1,
    'YellowRight': 1,
    'YellowStraight': 1,
    'YellowStraightLeft': 1,
    'YellowStraightRight': 1,
    # Green variants
    'Green': 2,
    'GreenLeft': 2,
    'GreenRight': 2,
    'GreenStraight': 2,
    'GreenStraightLeft': 2,
    'GreenStraightRight': 2,
}

import matplotlib.pyplot as plt

def debug_save_image_with_boxes(img_path, yolo_label_path, save_path):
    """Plot image with YOLO boxes and save to disk (server-friendly)."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Read YOLO labels
    if not os.path.exists(yolo_label_path):
        print(f"No labels found: {yolo_label_path}")
        return

    boxes = []
    with open(yolo_label_path, "r") as f:
        for line in f.readlines():
            class_id, xc, yc, bw, bh = map(float, line.split())
            # Convert normalized → pixel coords
            xc *= w
            yc *= h
            bw *= w
            bh *= h
            x1 = int(xc - bw/2)
            y1 = int(yc - bh/2)
            x2 = int(xc + bw/2)
            y2 = int(yc + bh/2)
            boxes.append((class_id, x1, y1, x2, y2))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")

    for class_id, x1, y1, x2, y2 in boxes:
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        )
        ax.text(x1, y1 - 3, CLASS_NAMES[int(class_id)], fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved debug image to: {save_path}")


def create_data_yaml(output_root):
    """Create YOLO data.yaml configuration file"""
    data_yaml = {
        'path': output_root,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    yaml_path = os.path.join(output_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml at: {yaml_path}")

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert BSTLD bbox format to YOLO format
    BSTLD format: [x_min, y_min, x_max, y_max] in pixels
    YOLO format: [x_center, y_center, width, height] normalized [0-1]
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def read_bstld_yaml(yaml_path):
    """Read BSTLD YAML annotation file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

    
# ---------- Add this helper near the top of your script ----------
def resolve_image_path(raw_path, bstld_root):
    """
    Given a raw path string from YAML and the bstld_root,
    try a list of candidate locations and return the first that exists.
    Returns absolute path or None if not found.
    """
    if not raw_path:
        return None

    # If already absolute and exists, return it
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path

    # Normalize tilde / leading ./ etc
    normalized = raw_path.lstrip("./")

    candidates = []

    # If path looks like "rgb/..." or "./rgb/..."
    candidates.append(os.path.join(bstld_root, normalized))

    # Plain basename (search for file directly under bstld_root/test or bstld_root/train or bstld_root)
    basename = os.path.basename(raw_path)
    candidates.extend([
        os.path.join(bstld_root, basename),
        os.path.join(bstld_root, "test", basename),
        os.path.join(bstld_root, "train", basename),
        os.path.join(bstld_root, "rgb", "test", basename),
        os.path.join(bstld_root, "rgb", "train", basename),
        os.path.join(bstld_root, "rgb", basename),
    ])

    # Also try raw_path appended to bstld_root (in case YAML is relative but without rgb/)
    candidates.append(os.path.join(bstld_root, raw_path))

    # De-duplicate while preserving order
    seen = set()
    dedup = []
    for c in candidates:
        cnorm = os.path.normpath(c)
        if cnorm not in seen:
            seen.add(cnorm)
            dedup.append(cnorm)

    for c in dedup:
        if os.path.exists(c):
            return c

    # Not found
    return None

# ---------- Replace process_split with this version ----------
def process_split(images_data, bstld_root, output_root, split_name):
    """Process a single split (train/val/test) with robust path resolution"""
    
    processed_count = 0
    skipped_count = 0
    total_boxes = 0
    class_counts = {label: 0 for label in CLASS_NAMES}
    label_issues = []
    
    for img_idx, img_info in enumerate(images_data):
        # Get image path (raw from YAML)
        raw_img_path = img_info.get('path', '')
        if not raw_img_path:
            skipped_count += 1
            continue

        # Resolve to a real local path
        full_img_path = resolve_image_path(raw_img_path, bstld_root)
        if full_img_path is None:
            print(f"  Warning: Image not found (resolved): {raw_img_path}")
            skipped_count += 1
            continue

        # Read image to get dimensions
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"  Warning: Cannot read image: {full_img_path}")
            skipped_count += 1
            continue
            
        img_height, img_width = img.shape[:2]
        
        # Get image filename
        img_filename = os.path.basename(full_img_path)
        img_name = os.path.splitext(img_filename)[0]
        
        # Copy image to output
        output_img_path = os.path.join(output_root, 'images', split_name, img_filename)
        shutil.copy2(full_img_path, output_img_path)
        
        # Convert annotations to YOLO format
        boxes = img_info.get('boxes', [])
        yolo_annotations = []
        
        # Debug first image's boxes
        if img_idx == 0 and len(boxes) > 0:
            print(f"\n  DEBUG - First image box structure:")
            print(f"  Image: {img_filename}")
            print(f"  Number of boxes: {len(boxes)}")
            print(f"  First box keys: {boxes[0].keys()}")
            print(f"  First box: {boxes[0]}")
        
        for box_idx, box in enumerate(boxes):
            # Try different possible label keys
            label = None
            for key in ['label', 'class', 'category', 'occluded_state']:
                if key in box:
                    label = box[key]
                    if img_idx == 0 and box_idx == 0:
                        print(f"  Found label in key '{key}': {label}")
                    break
            
            if label is None:
                if (img_idx, box_idx) not in [(i, j) for i, j in label_issues[:5]]:
                    label_issues.append((img_idx, box_idx))
                    if len(label_issues) <= 5:
                        print(f"  Warning: No label found in box {box_idx} of image {img_idx}")
                        print(f"    Box keys: {box.keys()}")
                        print(f"    Box: {box}")
                continue
            
            # Keep original case - BSTLD uses capitalized labels
            label = str(label)
            
            if label not in CLASS_MAPPING:
                # Try to add it to mapping dynamically based on color prefix
                label_lower = label.lower()
                if 'red' in label_lower:
                    class_id = 0
                    print(f"  Note: Mapping unknown red variant '{label}' to class 0 (red)")
                elif 'yellow' in label_lower:
                    class_id = 1
                    print(f"  Note: Mapping unknown yellow variant '{label}' to class 1 (yellow)")
                elif 'green' in label_lower:
                    class_id = 2
                    print(f"  Note: Mapping unknown green variant '{label}' to class 2 (green)")
                elif 'off' in label_lower:
                    class_id = 3
                    print(f"  Note: Mapping unknown off variant '{label}' to class 3 (off)")
                else:
                    if len(label_issues) <= 10:
                        print(f"  Warning: Unknown label '{label}' in {img_filename}, skipping")
                    continue
            else:
                class_id = CLASS_MAPPING[label]
            
            # Track class for statistics
            class_name = CLASS_NAMES[class_id]
            class_counts[class_name] += 1
            
            # Get bbox coordinates
            x_min = box.get('x_min', 0)
            y_min = box.get('y_min', 0)
            x_max = box.get('x_max', 0)
            y_max = box.get('y_max', 0)
            
            # Validate coordinates
            if x_max <= x_min or y_max <= y_min:
                print(f"  Warning: Invalid box coordinates in {img_filename}: {box}")
                continue
            
            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(
                [x_min, y_min, x_max, y_max], img_width, img_height
            )
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            total_boxes += 1
        
        # Write YOLO label file (even if empty, to track images without labels)
        output_label_path = os.path.join(output_root, 'labels', split_name, f"{img_name}.txt")
        if yolo_annotations:
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        else:
            # Create empty file for images without labels
            with open(output_label_path, 'w') as f:
                pass
        
        processed_count += 1
    
    print(f"\n  {split_name} Summary:")
    print(f"    Processed: {processed_count} images")
    print(f"    Skipped: {skipped_count} images")
    print(f"    Total boxes: {total_boxes}")
    print(f"    Class distribution:")
    for label in CLASS_NAMES:
        count = class_counts[label]
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"      {label}: {count} ({percentage:.1f}%)")
    
    if len(label_issues) > 5:
        print(f"    Note: {len(label_issues)} total boxes had label issues")

# ---------- Replace convert_bstld_to_yolo's loop so test is processed ----------
def convert_bstld_to_yolo(bstld_root, output_root):
    """
    Convert BSTLD dataset to YOLO format (updated: actually process test set + robust path resolution)
    """
    print("Converting BSTLD to YOLO format...")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        Path(output_root, 'images', split).mkdir(parents=True, exist_ok=True)
        Path(output_root, 'labels', split).mkdir(parents=True, exist_ok=True)
    
    # Process train and test sets
    for split in ['train', 'test']:
        yaml_file = os.path.join(bstld_root, f'{split}.yaml')
        
        if not os.path.exists(yaml_file):
            print(f"Warning: {yaml_file} not found, skipping {split} set")
            continue
            
        print(f"Processing {split} set...")
        data = read_bstld_yaml(yaml_file)
        
        images_data = data if isinstance(data, list) else data.get('images', [])
        
        if split == 'train':
            random.shuffle(images_data)
            split_idx = int(0.8 * len(images_data))
            train_data = images_data[:split_idx]
            val_data = images_data[split_idx:]
            
            process_split(train_data, bstld_root, output_root, 'train')
            process_split(val_data, bstld_root, output_root, 'val')
        else:
            # IMPORTANT: actually process the test set
            process_split(images_data, bstld_root, output_root, 'test')
    
    # Create data.yaml
    create_data_yaml(output_root)
    
    print(f"Conversion complete! YOLO dataset saved to: {output_root}")




print("Converting BSTLD to YOLO format...")
convert_bstld_to_yolo(BSTLD_ROOT, OUTPUT_ROOT)
