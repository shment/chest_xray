import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from skimage import io
from skimage.color import rgb2gray
import timm
import streamlit as st
import gdown
import os

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = timm.create_model('tf_efficientnetv2_b0', pretrained=True, in_chans=1)
        self.fc = nn.Linear(1000, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

if not os.path.exists('chest_xray_model.ckpt'):
    url = 'https://drive.google.com/uc?id=1jzDt06D3EJCcTul7PLDTSARLgN1uhlw7'
    gdown.download(url, 'chest_xray_model.ckpt', quiet=False)

model = EfficientNet()
param_path = 'chest_xray_model.ckpt'
model.load_state_dict(torch.load(param_path, map_location=torch.device('cpu')))
model.eval() 
st.write("## Chest X-Ray Pneumonia Detector")
upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if upload is not None:
    img = io.imread(upload)
    if len(img.shape) > 2: 
        img = rgb2gray(img)
        
    st.write("Original Image :camera:")
    st.image(img)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    img = transform(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = img.float()
    pred = model(img)
    int_to_labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
    pred = int_to_labels[torch.argmax(pred, axis=1).item()]
    st.write("Prediction (NORMAL or PNEUMONIA)")
    st.write(pred)