from ultralytics import YOLO

import cv2


model_path = '/home/ankit/Downloads/The-Perception-Saga/YOLO Model/runs/segment/train/weights/last.pt'

image_path = '/home/ankit/Downloads/washburn pics/1698439940027.png'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)
results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.cpu().numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./outputwashburn.png', mask)
