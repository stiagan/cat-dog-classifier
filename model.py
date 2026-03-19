import torch.nn as nn
from torchvision import models

def get_model():

    model = models.resnet50(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)

    return model
