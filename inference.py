import torch
import numpy as np
from PIL import Image, ImageOps
from model.cnn import CNN


def load_model(model_path="model.pth"):
    model = CNN(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        predicted = torch.argmax(outputs, dim=1).item()
    return predicted
