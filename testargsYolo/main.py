from ultralytics import YOLO

import supervision as sv
import cv2
import argparse


def Detect(mode, filename):
    model = YOLO(mode)
    file = filename

    image = cv2.imread(file)
    results = model(image)[0]
    detections = sv.Detections.from_yolov8(results)
    classes = model.model.names
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]
    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
        labels=labels
    )
    cv2.imshow('img', annotated_frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    required=True,
    help="Path to model",
    type=str
)
parser.add_argument(
    '--filename',
    required=True,
    help="Path to source",
    type=str
)

args = parser.parse_args()
if __name__ == "__main__":
    Detect(args.model, args.filename)
