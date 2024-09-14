import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the custom-trained model
model = YOLO('weapondetectionmodel2.pt')

# Load an image
image_path = 'trial/pistol1.jpg'
img = cv2.imread(image_path)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run the model on the image
results = model.predict(image_path)

# Get the results
boxes = results[0].boxes[0].xyxy.cpu().numpy()  # Get the coordinates of the bounding boxes
labels = results[0].boxes[0].cls.cpu().numpy()  # Get the labels

# Get class names from the model
class_names = results[0].names

# Draw bounding boxes and labels on the image
for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = box
    cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image_rgb, class_names[label], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Display the image with the detections
plt.imshow(image_rgb)
plt.axis('off')
plt.show()