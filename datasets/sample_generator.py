import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from .idd_schema import IDDDetectionClass, IDDSegmentationClass, BoundingBox2D, IDDImageSample

def generate_mock_idd_dataset(base_data_dir: str, num_train: int = 30, num_val: int = 8, num_test: int = 8):
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }
    classes = list(IDDDetectionClass)
    img_w, img_h = 640, 360

    for split_name, count in splits.items():
        split_img_dir = os.path.join(base_data_dir, split_name, "images")
        split_anno_dir = os.path.join(base_data_dir, split_name, "annotations")
        split_mask_dir = os.path.join(base_data_dir, split_name, "masks")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_anno_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        manifest = []

        for i in range(count):
            img_id = f"idd_{split_name}_{i:04d}"
            img_path = os.path.join(split_img_dir, f"{img_id}.png")
            mask_path = os.path.join(split_mask_dir, f"{img_id}_mask.png")
            json_path = os.path.join(split_anno_dir, f"{img_id}.json")

            img = Image.new("RGB", (img_w, img_h), color=(140, 160, 180))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, int(img_h * 0.45), img_w, img_h], fill=(45, 75, 45))
            draw.polygon([
                (int(img_w * 0.42), int(img_h * 0.45)),
                (int(img_w * 0.58), int(img_h * 0.45)),
                (img_w - 60, img_h),
                (60, img_h)
            ], fill=(70, 75, 80))

            mask = np.full((img_h, img_w), IDDSegmentationClass.ROADSIDE.value, dtype=np.uint8)
            for y in range(int(img_h * 0.45), img_h):
                alpha = (y - int(img_h * 0.45)) / (img_h - int(img_h * 0.45))
                x_left = int(img_w * 0.42 - alpha * (img_w * 0.42 - 60))
                x_right = int(img_w * 0.58 + alpha * (img_w - 60 - img_w * 0.58))
                mask[y, max(0, x_left):min(img_w, x_right)] = IDDSegmentationClass.ROAD.value

            num_objects = random.randint(1, 3)
            boxes = []

            for obj_idx in range(num_objects):
                cls = random.choice(classes)
                obj_y = random.randint(int(img_h * 0.52), img_h - 60)
                obj_x = random.randint(int(img_w * 0.25), int(img_w * 0.70))
                box_w = random.randint(35, 70)
                box_h = random.randint(40, 80)

                xmin = max(0, obj_x)
                ymin = max(0, obj_y)
                xmax = min(img_w - 1, xmin + box_w)
                ymax = min(img_h - 1, ymin + box_h)

                draw.rectangle([xmin, ymin, xmax, ymax], fill=(200, 100, 50), outline=(0, 0, 0))

                seg_label = IDDSegmentationClass.VEHICLES.value
                if cls == IDDDetectionClass.PEDESTRIAN:
                    seg_label = IDDSegmentationClass.PEDESTRIANS.value
                elif cls == IDDDetectionClass.ANIMAL:
                    seg_label = IDDSegmentationClass.OBSTACLES.value
                mask[ymin:ymax, xmin:xmax] = seg_label

                boxes.append(BoundingBox2D(
                    xmin=float(xmin),
                    ymin=float(ymin),
                    xmax=float(xmax),
                    ymax=float(ymax),
                    label=cls
                ))

            img.save(img_path)
            Image.fromarray(mask).save(mask_path)

            sample = IDDImageSample(
                image_id=img_id,
                file_path=os.path.relpath(img_path, base_data_dir),
                width=img_w,
                height=img_h,
                boxes=boxes,
                segmentation_mask_path=os.path.relpath(mask_path, base_data_dir)
            )

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(sample.model_dump_json(indent=2))

            manifest.append(sample.model_dump())

        manifest_path = os.path.join(base_data_dir, f"{split_name}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"IDD synthetic dataset generated at: {base_data_dir}")
