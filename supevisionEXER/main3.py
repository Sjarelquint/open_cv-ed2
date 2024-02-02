import wget

# url= 'https://docs.google.com/uc?export=download&id=1vVrEVMxucHgqGd7vAa501ASojbeGPhIr'
# filename = wget.download(url)
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np

polygon = np.array([
    [540, 985],
    [1620, 985],
    [2160, 1920],
    [1620, 2855],
    [540, 2855],
    [0, 1920]
])

gen = sv.get_video_frames_generator('market-square.mp4', stride=5)
video_info = sv.VideoInfo.from_video_path('market-square.mp4')
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

model = YOLO('yolov8n.pt')
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

for frame in gen:
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    mask = zone.trigger(detections=detections)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5) & mask]

    labels = [
        f" {confidence:0.2f}"
        for _, _, confidence, _, _
        in detections
    ]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    # frame = sv.draw_polygon(scene=frame, polygon=polygon, color=sv.Color.red(), thickness=6)
    frame = zone_annotator.annotate(scene=frame)

    cv2.imshow('img', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
