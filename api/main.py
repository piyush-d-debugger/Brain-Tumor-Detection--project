from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import os

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "saved_models", "1.1"))

MODEL = tf.saved_model.load(MODEL_PATH)

INFER = MODEL.signatures["serving_default"]


CLASS_NAMES = ["no", "pred", "yes"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_imagefile(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((630, 630), Image.Resampling.LANCZOS)
    
    
    image_array = np.array(image)
    
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    img_batch = np.expand_dims(image, 0).astype(np.float32)

    
    tensor = tf.constant(img_batch)
    
    predictions_dict = INFER(tensor)
    predictions = predictions_dict['dense_1'].numpy()

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)