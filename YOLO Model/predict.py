from ultralytics import YOLO

import cv2


model_path = '/home/ankit/Downloads/WindowYoloV8/runs/segment/train2/weights/last.pt'

image_path = '/home/ankit/Downloads/1698439921939.png'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)
results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.cpu().numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output2.png', mask)
