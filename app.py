import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from model.cnn import CNN

# Load trained model
model = CNN(in_channels=3, num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

app = FastAPI()

# Allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS/JS/images)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")  # Debug
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_image = pil_image.resize((64, 64))

        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            prediction = torch.argmax(outputs, dim=1).item()

        label = "Cat" if prediction == 0 else "Dog"
        print(f"Prediction: {label}")  # Debug

        return {"prediction": label}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# To run the server:
# uvicorn app:app --reload
# Then access http://127.0.0.1:8000/