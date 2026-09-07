import cv2
from ultralytics import YOLO

from fall_detector import FallDetector

model = YOLO("yolo11n-pose.pt")
detector = FallDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)
    result = results[0]

    state = "NO_PERSON"

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        keypoints = result.keypoints.xy[0].cpu().numpy()

        state = detector.update(
            keypoints=keypoints,
            frame_height=frame.shape[0]
        )

    annotated_frame = result.plot()

    cv2.putText(
        annotated_frame,
        state,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Fall Detection MVP", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()