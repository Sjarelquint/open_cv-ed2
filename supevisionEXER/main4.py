import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
from supervision.geometry.core import Point

line = np.array([
    [112, 1664], [3456, 1580]
])
point1 = Point(112, 1664)
point2 = Point(3456, 1580)
gen = sv.get_video_frames_generator('vehicles.mp4')
frame = next(iter(gen))
cv2.imwrite('frame2.png', frame)
model = YOLO('yolov8n.pt')
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
tracker = sv.ByteTrack()
zone = sv.LineZone(point1, point2 )
line_annotator = sv.LineZoneAnnotator( color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

for frame in gen:
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[(detections.class_id == 2) & (detections.confidence > 0.5)]
    detections = tracker.update_with_detections(detections)
    zone.trigger(detections=detections)
    labels = [

        f" {tracker_id} "
        for tracker_id
        in detections.tracker_id

    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = line_annotator.annotate(frame,zone,)
    cv2.imshow('img', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
