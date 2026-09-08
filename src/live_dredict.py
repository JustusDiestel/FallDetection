from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from fall_model import FallClassifier
from preprocess_keypoints import normalize_sequence


WINDOW_SIZE = 30

pose_model = YOLO("yolo11n-pose.pt")

fall_model = FallClassifier()
fall_model.load_state_dict(
    torch.load(
        "fall_classifier.pt",
        map_location="cpu"
    )
)
fall_model.eval()


sequence_buffer = deque(
    maxlen=WINDOW_SIZE
)


cap = cv2.VideoCapture(0)


while True:

    success, frame = cap.read()

    if not success:
        break


    result = pose_model(
        frame,
        verbose=False
    )[0]


    if (
        result.keypoints is None
        or result.keypoints.xy.shape[0] == 0
    ):
        keypoints = None

    else:
        keypoints = (
            result.keypoints.xy[0]
            .cpu()
            .numpy()
        )


    if keypoints is not None:

        sequence_buffer.append(
            keypoints
        )

    elif len(sequence_buffer) > 0:

        sequence_buffer.append(
            sequence_buffer[-1]
        )


    prediction_text = "WAITING"


    if len(sequence_buffer) == WINDOW_SIZE:

        sequence = np.array(
            sequence_buffer,
            dtype=np.float32
        )

        sequence = normalize_sequence(
            sequence
        )

        x = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        x = x.unsqueeze(0)


        with torch.no_grad():

            output = fall_model(x)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            fall_probability = probabilities[
                0, 1
            ].item()


        if prediction == 1:

            prediction_text = (
                f"FALL {fall_probability:.2f}"
            )

        else:

            prediction_text = (
                f"NORMAL {fall_probability:.2f}"
            )


    cv2.putText(
        frame,
        prediction_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )


    cv2.imshow(
        "Fall Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()