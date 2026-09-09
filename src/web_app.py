from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from src.fall_model import FallClassifier
from src.preprocess_keypoints import extract_features


WINDOW_SIZE = 30

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


pose_model = YOLO("src/yolo11n-pose.pt")

fall_model = FallClassifier()
fall_model.load_state_dict(
    torch.load(
        "src/fall_classifier.pt",
        map_location="cpu"
    )
)
fall_model.eval()


sequence_buffer = deque(
    maxlen=WINDOW_SIZE
)


@app.get("/")
def index():
    return FileResponse(
        Path("static/index.html")
    )


@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):

    image_bytes = await image.read()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    result = pose_model(
        frame,
        verbose=False
    )[0]


    if (
        result.keypoints is None
        or result.keypoints.xy.shape[0] == 0
    ):
        return {
            "person_detected": False,
            "status": "NO_PERSON",
            "fall_probability": 0.0
        }


    keypoints = (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )
    keypoints_list = keypoints.tolist()

    sequence_buffer.append(
        keypoints
    )


    if len(sequence_buffer) < WINDOW_SIZE:
        return {
            "person_detected": True,
            "status": "WAITING",
            "fall_probability": 0.0,
            "keypoints": keypoints_list
        }


    sequence = np.array(
        sequence_buffer,
        dtype=np.float32
    )

    sequence = extract_features(
        sequence
    )


    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)


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


    status = (
        "FALL"
        if prediction == 1
        else "NORMAL"
    )

    return {
        "person_detected": True,
        "status": status,
        "fall_probability": fall_probability,
        "keypoints": keypoints_list
    }