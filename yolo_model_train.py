import os
from roboflow import Roboflow
from ultralytics import YOLO

import nms_patch

rf = Roboflow(api_key="O88VXaV6eS96GnoV3BwU")
project = rf.workspace("final-project-mvwge").project("box_detector-8l7t1")
version = project.version(2)
dataset = version.download("yolov8")

model = YOLO('yolov8n.pt')

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    project='runs/train',
    name='package_detection_yolo',
    amp=False,
    verbose=True,
    patience=10,
    workers=2,
    cache=False,
    single_cls=False,
    rect=False,
    cos_lr=False,
    close_mosaic=10,
    resume=False,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True,
    save=True,
    save_period=-1,
    plots=True,
    deterministic=True,
    seed=0,
    exist_ok=False,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.01,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    label_smoothing=0.0,
    nbs=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
)

print("Training complete")
print(f"Model saved to: runs/train/package_detection_yolo/weights/best.pt")