from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

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



MODEL = tf.keras.models.load_model("../saved_models/1.1")
CLASS_NAMES = ["no", "pred", "yes"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def resize_image(data) -> np.ndarray:
    new_width = 630
    new_height = 630
    image = Image.open(BytesIO(data))
    width, height = image.size
    ratio = min(new_width / width, new_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    return image_array

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = resize_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index]
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

