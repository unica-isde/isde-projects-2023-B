import json
from typing import Dict, List, Annotated
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
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
import cv2
import numpy as np


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


