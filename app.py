# Updated app.py
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import torch
import numpy as np
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import base64
from aruco_finder_measure import findArucoMarkers

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <head>
            <title>Cucumber App</title>
        </head>
        <body>
            <h1>Upload or Take a Photo</h1>
            <form id="upload-form" enctype="multipart/form-data" method="post" action="/upload">
                <label>Plot Number: <input type="text" name="plot" required></label><br><br>
                <video id="video" width="300" autoplay></video><br>
                <button type="button" onclick="takePhoto()">Take Photo</button><br><br>
                <input type="file" name="file" id="file">
                <input type="submit" value="Upload">
            </form>

            <form method="get" action="/reset">
                <button type="submit">Reset CSV</button>
            </form><br>

            <canvas id="canvas" style="display:none;"></canvas>
            <div id="preview">
                <h3>Preview</h3>
                <img id="preview-img" style="max-width:100%; display:none;" />
            </div>

            <script>
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                const previewImg = document.getElementById('preview-img');

                navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                    };
                });

                function takePhoto() {
                    const context = canvas.getContext('2d');
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);

                    const dataUrl = canvas.toDataURL("image/jpeg");
                    previewImg.src = dataUrl;
                    previewImg.style.display = "block";

                    canvas.toBlob(blob => {
                        const fileInput = document.getElementById('file');
                        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        fileInput.files = dataTransfer.files;
                    }, "image/jpeg");
                }
            </script>
        </body>
    </html>
    """

os.makedirs("uploads", exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "model_checkpoints/cucumber_models/model_final.pth"
predictor = DefaultPredictor(cfg)

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_cucumber_dataset_train", lambda: [])
MetadataCatalog.get("my_cucumber_dataset_train").thing_classes = ["Cucumber"]

CSV_FILE = "results.csv"
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["plot", "id", "filename", "class", "mask_area_px", "major_axis_px", "minor_axis_px", "mask_area_cm2", "major_axis_cm", "minor_axis_cm", "size_class"]).to_csv(CSV_FILE, index=False)

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...), plot: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pixels_per_mm = findArucoMarkers(image)
    pixels_per_cm = pixels_per_mm * 10

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    metadata = MetadataCatalog.get("my_cucumber_dataset_train")
    class_names = metadata.get("thing_classes", [])

    csv_rows = []

    def get_size_class(width_in_inches):
        if width_in_inches < 1 + 1/16:
            return "1"
        elif width_in_inches < 1.25:
            return "2A"
        elif width_in_inches < 1.5:
            return "2B"
        elif width_in_inches < 1.75:
            return "3A"
        elif width_in_inches < 2:
            return "3B"
        elif width_in_inches < 2 + 2/16:
            return "4"
        else:
            return "Oversize"

    for i, (mask, cls) in enumerate(zip(masks, classes)):
        class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"

        if class_name != "Cucumber":
            continue

        mask_uint8 = (mask * 255).astype(np.uint8)
        area = np.sum(mask)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        major_axis = minor_axis = 0

        if len(contours) > 0 and len(contours[0]) >= 5:
            ellipse = cv2.fitEllipse(contours[0])
            (_, _), (MA, ma), _ = ellipse
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
        else:
            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0 and len(y_indices) > 0:
                width = x_indices.max() - x_indices.min()
                height = y_indices.max() - y_indices.min()
                major_axis = max(width, height)
                minor_axis = min(width, height)

        area_cm2 = area / (pixels_per_cm ** 2) if pixels_per_cm > 0 else 0
        major_cm = major_axis / pixels_per_cm if pixels_per_cm > 0 else 0
        minor_cm = minor_axis / pixels_per_cm if pixels_per_cm > 0 else 0

        # Convert to inches for classification (1 inch = 2.54 cm)
        minor_in = minor_cm / 2.54
        size_class = get_size_class(minor_in)

        csv_rows.append({
            "plot": plot,
            "id": i,
            "filename": file.filename,
            "class": class_name,
            "mask_area_px": int(area),
            "major_axis_px": float(major_axis),
            "minor_axis_px": float(minor_axis),
            "mask_area_cm2": round(area_cm2, 2),
            "major_axis_cm": round(major_cm, 2),
            "minor_axis_cm": round(minor_cm, 2),
            "size_class": size_class
        })

    df = pd.DataFrame(csv_rows)
    df.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))

    _, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return f"""
    <html>
        <body>
            <h2>Detections Complete</h2>
            <img src='data:image/png;base64,{encoded_image}' style='max-width:100%;'/>
            <br><a href="/download">Download CSV</a>
            <br><a href="/">Go Back</a>
        </body>
    </html>
    """

@app.get("/download")
async def download_csv():
    return FileResponse(CSV_FILE, media_type='text/csv', filename="results.csv")

@app.get("/reset")
async def reset_csv():
    df = pd.DataFrame(columns=["plot", "id", "filename", "class", "mask_area_px", "major_axis_px", "minor_axis_px", "mask_area_cm2", "major_axis_cm", "minor_axis_cm", "size_class"])
    df.to_csv(CSV_FILE, index=False)
    return {"message": "CSV file reset successfully."}
