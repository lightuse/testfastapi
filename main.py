from fastapi import FastAPI
from fastapi import APIRouter, File, UploadFile
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://pytorch-cpu.herokuapp.com",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
from os.path import join, dirname, realpath
from torchvision import models, transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import timm
from PIL import Image

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/health')
def health():
    return {
        'message': 'ok'
    }

@app.post('/post')
def simple_post(param: str):
    return {
        'message': f'You posted `{param}`!'
    }

#ラベル数
n_class = 6
#モデル名
model_name = 'tf_efficientnetv2_s_in21ft1k'
model_parameter = 'weights_fine_tuning_effcientv2.pth'
#画像サイズ
image_size = 160
class_names = ['1_建物から庭', '2_外の庭', '5_お菓子', '6_洋館内部', '7_建物屋敷', '9_その他']

@app.post('/inference')
def inference(file: UploadFile = File(...)):
    path = file.file
    img = Image.open(path).convert('RGB')
    data_config = timm.data.resolve_data_config({}, model=model_name, verbose=True)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(data_config["mean"], data_config["std"]),
    ])
    device = torch.device("cpu")
    inputs = transform(img)
    inputs = inputs.unsqueeze(0).to(device)
    model = EfficientNet_b0(n_class)
    path = "./"
    path = os.path.join(path, model_parameter)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    outputs = model(inputs)
    batch_probs = F.softmax(outputs, dim=1)
    batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
    for probs, indices in zip(batch_probs, batch_indices):
        for k in range(n_class):
            print(f"Top-{k + 1} {class_names[indices[k]]} {probs[k]:.2%}")
            a = class_names[indices[k]]
            b = probs[k].item()
            return {"result":"OK", "file":a, "probality":b}

class EfficientNet_b0(nn.Module):
    def __init__(self, n_out):
        super(EfficientNet_b0, self).__init__()
        #モデルの定義
        self.effnet = timm.create_model(model_name, pretrained=True)
        #最終層の再定義
        self.effnet.classifier = nn.Linear(1280, n_out)

    def forward(self, x):
        return self.effnet(x)