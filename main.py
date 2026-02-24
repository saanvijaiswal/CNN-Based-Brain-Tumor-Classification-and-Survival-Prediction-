from fastapi import FastAPI, File, UploadFile 
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf 
import numpy as np 
from PIL import Image
import io 
import json

app = FastAPI()
#allow frontend later
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("tumor_model_finalmaolmao.h5")

# Load class labels
with open("labels.json", "r") as f:
    class_names = json.load(f)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def home():
    return {"message": "Brain Tumor API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[str(predicted_index)]
    confidence = float(np.max(prediction))
    

    return {
        "tumor_type": predicted_label,
        "confidence": f"{round(confidence * 100, 2)}%"
    }