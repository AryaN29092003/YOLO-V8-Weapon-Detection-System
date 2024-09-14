import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("weapondetectionmodel2.pt")

results = model.predict("trial\\pistol1.jpg",save =True)
# for r in results:
        
#         #annotator = Annotator(img)
        
#         boxes = r.boxes
#         for box in boxes:
            
#             b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
#             c = box.cls
#             #annotator.box_label(b, model.names[int(c)])
# # x1 = b[0][0]
# # y1 = b[0][1]
# # x2 = b[0][2]
# # y2 = b[0][3]

# boxes = results[0].boxes[0].xyxy.cpu().numpy()
print(results)
# print("c: ", c)
