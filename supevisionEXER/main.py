import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
colors = sv.ColorPalette.default()
#enter
zoneIn=[np.array([
[354, 556],[330, 680],[686, 760],[722, 628]
]),
np.array([
[1154, 388],[1146, 528],[1594, 532],[1598, 392]
]),
np.array([
[750, 56],[754, 264],[822, 260],[830, 48]
])]
#leave
zoneOut=[np.array([
[462, 428],[446, 540],[690, 552],[690, 412]
]),
np.array([
[1186, 628],[1234, 748],[1462, 656],[1442, 544]
]),
np.array([
[1078, 108],[1070, 328],[1154, 336],[1154, 108]
])]
#stay
stay=np.array([
[802, 396],[746, 496],[706, 608],[826, 800],[914, 872],[1106, 664],[1154, 512],[1090, 368],[841, 360]
])
box_annotator=sv.BoxAnnotator()
model = YOLO('traffic_analysis.pt')
video_path='C:/Users/azpow/PycharmProjects/trafficAnalysis/Data/traffic_analysis.mov'
gen = sv.get_video_frames_generator(video_path,stride=20)
video_info = sv.VideoInfo.from_video_path(video_path=video_path)
zones = [
    sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=video_info.resolution_wh
    )
    for polygon
    in zoneIn
]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(index),
        thickness=4,
        text_thickness=8,
        text_scale=4
    )
    for index, zone
    in enumerate(zones)
]


cv2.imwrite('frame1.png',next(iter(gen)))
img=cv2.imread('frame1.png')
cv2.imshow('frame1',img)
for frame in gen:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_frame=box_annotator.annotate(frame,detections=detections)
    for zone, zone_annotator in zip(zones, zone_annotators):
        frame = zone_annotator.annotate(scene=frame)

    cv2.imshow('img', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
