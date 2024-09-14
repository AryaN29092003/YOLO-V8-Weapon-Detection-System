from ultralytics import YOLO
import cv2
import time
import supervision as sv


def main() -> None:
    model = YOLO("weapondetectionmodel2.pt") # detects small gun, fire, big gun

    cap = cv2.VideoCapture("video.avi")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    avg_inf_time = 0
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_weapon_yolov8.avi', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            results = model(frame) 
            end_time = time.time()

            inference_time = end_time - start_time
            avg_inf_time += inference_time
            frame_count += 1

            frames_to_skip = int(inference_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_to_skip)
            
            detections = sv.Detections.from_ultralytics(results[0])
            labels = [model.model.names[class_id] for class_id in detections.class_id]
            annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            cv2.imshow('Frame', annotated_image)
            for _ in range(frames_to_skip + 1):
                out.write(annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    print(f"Average inference time: {avg_inf_time / frame_count}s")


if __name__ == '_main_':
    main()