const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const overlay = document.getElementById("overlay");
const overlayContext = overlay.getContext("2d");

const statusElement = document.getElementById("status");
const probabilityElement = document.getElementById("probability");


async function startCamera() {

    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    });

    video.srcObject = stream;
}


async function sendFrame() {

    if (video.videoWidth === 0) {
        return;
    }


    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;


    const context = canvas.getContext("2d");

    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );


    const blob = await new Promise(
        resolve => canvas.toBlob(
            resolve,
            "image/jpeg",
            0.7
        )
    );


    const formData = new FormData();

    formData.append(
        "image",
        blob,
        "frame.jpg"
    );


    const response = await fetch(
        "/predict",
        {
            method: "POST",
            body: formData
        }
    );


    const result = await response.json();
    drawKeypoints(result.keypoints);

    statusElement.textContent =
        result.status;


    probabilityElement.textContent =
        "Fall probability: " +
        (
            result.fall_probability * 100
        ).toFixed(1) +
        " %";
}


startCamera();


setInterval(
    sendFrame,
    100
);

function drawKeypoints(keypoints) {

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    overlayContext.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    if (!keypoints) {
        return;
    }

    for (const [x, y] of keypoints) {

        overlayContext.beginPath();

        overlayContext.arc(
            x,
            y,
            5,
            0,
            Math.PI * 2
        );

        overlayContext.fillStyle = "red";
        overlayContext.fill();
    }
}