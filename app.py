import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

from models.model import get_model
from utils.transforms import val_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))

model.to(device)
model.eval()

classes = ["Cat", "Dog"]

def predict(image):

    image = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    return {
        classes[0]: float(probs[0][0]),
        classes[1]: float(probs[0][1])
    }

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Cat vs Dog Classifier"
)

interface.launch()
