import json
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from typing import Dict, List, Annotated
from fastapi import FastAPI, Request, HTTPException, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.forms.enhancement_form import EnhancementForm
from app.ml.classification_utils import classify_image
from PIL import ImageEnhance, Image
import base64
from io import BytesIO
from app.forms.histogram_form import HistogramForm
from app.utils import list_images
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import magic


app = FastAPI()
config = Configuration()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/info")
def info() -> Dict[str, List[str]]:
    """Returns a dictionary with the list of models and
    the list of available image files."""
    list_of_images = list_images()
    list_of_models = Configuration.models
    data = {"models": list_of_models, "images": list_of_images}
    return data


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """The home page of the service."""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )


@app.post("/classifications")
async def request_classification(request: Request):
    form = ClassificationForm(request)
    await form.load_data()
    image_id = form.image_id
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),
        },
    )



@app.get("/custom_classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "custom_classification_select.html",
        {"request": request, "models": Configuration.models},
    )


@app.post("/custom_classifications")
async def upload_file(file: UploadFile, request: Request):
    try:
        # Legge il contenuto del file
        file_content = await file.read()

        # Salva temporaneamente file nel server nella cartella "static" --> imagenet_subset
        original_path = f"app/static/imagenet_subset/{file.filename}"
        with open(original_path, "wb") as f:
            f.write(file_content)
        image_id = file.filename

        # Controlla se il file inserito è effettivamente un immagine
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(original_path)
        if not file_type.startswith("image"):
            os.remove(original_path)
            raise ValueError("Uploaded file is not an image")

        # Salva il file temporaneamente in un buffer
        buffer = BytesIO()
        img = Image.open(original_path)
        img.save(buffer, format="PNG")

        image_64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{image_64}"

        form = ClassificationForm(request)
        await form.load_data()
        model_id = form.model_id
        classification_scores = classify_image(model_id=model_id, img_id=image_id)

        # Rimuove il file da static
        os.remove(original_path)

        # Invia il file dal buffer
        return templates.TemplateResponse(
            "custom_classification_output.html",
            {
                "request": request,
                "image_id": data_url,
                "classification_scores": json.dumps(classification_scores),
            },
        )
    except Exception as e:
        return {"error": f"An error occurred during the file upload: {str(e)}"}

@app.get("/download_results", response_class=JSONResponse)
def download_results(classification_scores):
    results = json.loads(classification_scores)
    labels = [result[0] for result in results][::-1]
    scores = [result[1] for result in results][::-1]
    results_dict = {labels[0]: scores[0], labels[1]: scores[1], labels[2]: scores[2], labels[3]: scores[3], labels[4]: scores[4]}
    return JSONResponse(content=results_dict,
                        media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=results.json"})


#Implementation Issue 3
@app.get("/download_plot", response_class=StreamingResponse)
async def download_plot(classification_scores: str):
    results = json.loads(classification_scores)
    labels = [result[0] for result in results][::-1]
    scores = [result[1] for result in results][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        (63 / 255, 3 / 255, 85 / 255, 0.8),
        (6 / 255, 33 / 255, 108 / 255, 0.8),
        (121 / 255, 87 / 255, 3 / 255, 0.8),
        (117 / 255, 0 / 255, 20 / 255, 0.8),
        (26 / 255, 74 / 255, 4 / 255, 0.8)
    ]
    ax.barh(labels, scores, color=colors)
    ax.set_xlabel('Scores')
    ax.set_title('Top 5 Classification Scores')
    ax.grid()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close(fig)
    img = Image.open(img_buffer)
    img_byte = BytesIO()
    img.save(img_byte, format='PNG')
    img.close()
    img_byte.seek(0)
    return StreamingResponse(content=img_byte, media_type="image/png", headers={"Content-Disposition": "attachment; filename=plot.png"})

  
#Implementation Issue 2
@app.get("/enhancement")
def create_transformed_image(request: Request):
    return templates.TemplateResponse(
        "enhancement_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )


@app.post("/enhancement")
async def apply_transformation(request: Request):
    form = EnhancementForm(request)
    await form.load_data()

    image_id = form.image_id
    transformation_params = {
        "color": form.color,
        "brightness": form.brightness,
        "contrast": form.contrast,
        "sharpness": form.sharpness,
    }

    transformed_image_path = apply_image_transformation(image_id, transformation_params)

    return templates.TemplateResponse("enhancement_output.html",
                                      {"request": request,
                                       "color": form.color,
                                       "brightness": form.brightness,
                                       "contrast": form.contrast,
                                       "sharpness": form.sharpness,
                                       "image_id": image_id,
                                       "transformed_image_path": transformed_image_path
                                       })
def apply_image_transformation(image_id, params):
    try:
        image_path = f"app/static/imagenet_subset/{image_id}"
        img = Image.open(image_path)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(params["color"])
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(params["brightness"])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(params["contrast"])
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(params["sharpness"])


        buffer = BytesIO()
        img.save(buffer, format="PNG")

        image_64= base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{image_64}"

        return data_url

    except Exception as e:
        error_message = f"Error during image transformation: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)




# Implementation for Issue 1
@app.get("/image_histogram")
def create_histogram(request: Request):
    return templates.TemplateResponse(
        "histogram_select.html",
        {"request": request, "images": list_images()},
    )


@app.post("/image_histogram")
async def request_classification(request: Request):
    form = HistogramForm(request)
    await form.load_data()
    image_id = form.image_id

    # read image
    im = cv2.imread('app/static/imagenet_subset/'+image_id)
    # calculate mean value from RGB channels and flatten to 1D array
    vals = im.mean(axis=2).flatten()
    # calculate histogram
    histogram, bins = np.histogram(vals, range(257))

    return templates.TemplateResponse(
        "histogram_output.html",
        {
            "request": request,
            "image_id": image_id,
            "histogram": json.dumps(histogram.tolist()),
        },
    )
