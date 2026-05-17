from ultralytics import YOLO
import torch

model = YOLO("yolov8-seg.pt")

print(model.info())

# print(model.model)

print(model.task)

ckpt = torch.load(
    "yolov8-seg.pt",
    map_location="cpu",
    weights_only=False
)

print(ckpt.keys())

print(ckpt['train_args'])