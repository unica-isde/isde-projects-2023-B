import json
from typing import Dict, List
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import redis
from rq import Connection, Queue
from rq.job import Job
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images

from PIL import Image
import os

# ----------------------

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


# Issue No.4 Upload an image--------------------------------------------------------------------


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

        # Salva il file nel server nella cartella "uploads" --> imagenet_subset
        original_path = f"app/static/imagenet_subset/{file.filename}"
        with open(original_path, "wb") as f:
            f.write(file_content)

        # Controlla il formato dell'immagine e converte se necessario
        if not file.filename.upper().endswith('.JPEG'):
            file_name_without_extension = file.filename.rsplit('.', 1)[0]
            converted_path = f"app/static/imagenet_subset/{file_name_without_extension}.JPEG"

            with Image.open(original_path) as img:
                img.convert("RGB").save(converted_path, "JPEG")

            os.remove(original_path)  # Rimuove la vecchia copia

            # Aggiorna il nome del file con l'estensione .JPEG
            image_id = f"{file_name_without_extension}.JPEG"
        else:
            # Se il file è già in formato .JPEG, usa il nome originario
            image_id = file.filename

        form = ClassificationForm(request)
        await form.load_data()
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
    except Exception as e:
        return {"error": f"Si è verificato un errore durante l'upload del file: {str(e)}"}
