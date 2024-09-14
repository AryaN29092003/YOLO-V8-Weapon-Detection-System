import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

import numpy as np
from ultralytics import YOLO

class YOLOv8GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("YOLOv8 Model GUI")
        self.master.geometry("800x600")

        # Create frames for better organization
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(pady=10)

        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.pack(pady=10)

        # Create video output label
        self.video_output = tk.Label(self.top_frame)
        self.video_output.pack()

        # Create buttons
        self.camera_button = tk.Button(self.bottom_frame, text="Access Camera", command=self.access_camera)
        self.camera_button.grid(row=0, column=0, padx=10, pady=5)

        self.upload_video_button = tk.Button(self.bottom_frame, text="Upload Video", command=self.upload_video)
        self.upload_video_button.grid(row=0, column=1, padx=10, pady=5)

        self.upload_photo_button = tk.Button(self.bottom_frame, text="Upload Photo", command=self.upload_photo)
        self.upload_photo_button.grid(row=0, column=2, padx=10, pady=5)

        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.BOTTOM, padx=10, pady=10, anchor=tk.SE)

        self.submit_button = tk.Button(self.master, text="Submit", command=self.submit)
        self.submit_button.pack(side=tk.BOTTOM, padx=10, pady=10, anchor=tk.SE)

        # Initialize YOLOv8 model
        self.model = YOLO("weapondetectionmodel2.pt")

    def access_camera(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            # Display processed frame
            self.submit(frame)

            

        cap.release()
        cv2.destroyAllWindows()

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4",".avi")])
        if file_path:
            cap = cv2.VideoCapture(file_path)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                # Display processed frame
                self.submit(frame)

                

            cap.release()

    def upload_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                # Display processed frame
                frame = cv2.resize(frame, (640, 480))
                self.submit(frame)
                #self.display_frame(frame)
            else:
                print("Error: Could not load image")

    def clear(self):
        # Clear the video output label
        self.video_output.configure(image=None)
        self.video_output.image = None

    def submit(self,frame):
        # Implement model inference on uploaded image/video and display results
        stat,box,labels,class_names = self.detect_objects(frame)
        if(not stat):
            self.display_frame1(frame)
        else:
            self.display_frame(box,labels,class_names, frame)

    def detect_objects(self, frame):
        # Implement object detection using YOLOv8 model
        results=self.model.predict(frame, conf=0.3)
        class_names = results[0].names
        try:
            boxes = results[0].boxes[0].xyxy.cpu().numpy()  # Get the coordinates of the bounding boxes
            labels = results[0].boxes[0].cls.cpu().numpy()  # Get the labels
        except :
            return False,False,False,class_names
        
        return True,boxes, labels, class_names

    def display_frame(self ,box,labels,class_names , frame):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = box
        labels = labels
        class_names = class_names
        for box, label in zip(boxes,labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (150, 0, 150), 2)
            cv2.putText(frame_rgb, class_names[label], (int(x1)+ 10, int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 0, 150), 2)

        # Convert frame to ImageTk format
        image = Image.fromarray(frame_rgb)
        image_tk = ImageTk.PhotoImage(image)

        # Update video output label
        self.video_output.configure(image=image_tk)
        self.video_output.image = image_tk

    def display_frame1(self , frame):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to ImageTk format
        image = Image.fromarray(frame_rgb)
        image_tk = ImageTk.PhotoImage(image)

        # Update video output label
        self.video_output.configure(image=image_tk)
        self.video_output.image = image_tk

def main():
    root = tk.Tk()
    app = YOLOv8GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
