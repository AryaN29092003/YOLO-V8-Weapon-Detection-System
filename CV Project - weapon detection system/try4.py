import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt


model = YOLO('weapondetectionmodel2.pt')
# Function to process video frames
def process_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the model on the frame
    results = model.predict(frame)

    # Get the results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get the coordinates of the bounding boxes
    labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy().astype(int)]  # Get the labels

    # Draw bounding boxes and labels on the frame
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add coordinates as text alongside each bounding box
        coordinate_text = f"({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})"
        cv2.putText(image_rgb, coordinate_text, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_rgb

# Open the video file
video_path = 'D:\\arya study material\semVI\CV project\weapon detection pro\\trial\\video2.mp4'  # Update with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Create a video writer to save the processed video
output_video_path = '/trial'  # Update with your desired output video path
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame = process_frame(frame)


    plt.imshow(processed_frame)
    plt.axis('off')
    plt.show()
    out.write(processed_frame) 


    if cv2.waitKey(0) :
        break

cap.release()
out.release()
cv2.destroyAllWindows()