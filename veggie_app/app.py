import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 3. Define the model used for training
class VeggieNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 128 * 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Manually loading the class names to match the dataset
class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

#loading the model
device = "gpu" if torch.cuda.is_available() else "cpu"
model = VeggieNet(num_classes=len(class_names))
model.load_state_dict(torch.load("/home/royalbrothers/PyTorchProjects/models/veggie_net.pth", map_location=device))
model.eval()

#image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# prediction function
def predict(image):
    img = image.convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# gradio ui
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="🥕 Vegetable Image Classifier",
    description="Upload a vegetable image and the model will try to guess what it is!"
)

#launching the app
if __name__  == "__main__":
    interface.launch()