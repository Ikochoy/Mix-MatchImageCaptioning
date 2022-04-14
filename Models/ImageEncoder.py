import torch.nn as nn

import numpy as np
import torchvision.models as models 
import torch
from torchvision import transforms
from PIL import Image


class ImageEncoder(nn.Module):
  def __init__(self, choice, dropout=False, fine_tune=False):

    super(ImageEncoder, self).__init__()

    # self.input = CROPPED_WIDTH
    # Detect if we have a GPU available
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.trained = False

    if dropout:
        self.dropout = nn.Dropout(p=0.5)
    else:
        self.dropout = None

    if choice == "GoogLeNet":
      # GoogLeNet
      # google net would need to resize to be 299 insted of 224
      self.model_name = "GoogLeNet_Inception_v3"
      model_ft = models.inception_v3(pretrained=True, aux_logits=False)
      model_ft.fc = nn.Linear(self.model.fc.in_features,4096)
      self.model = model_ft.to(self.device)  
      self.linear = None
    

    elif choice == "AlexNet":
      self.model_name = "AlexNet"
      model_ft = models.alexnet(pretrained=True)

      layers = list(model_ft.children())[:-1]
      # remove classifier layer
      model_ft = torch.nn.Sequential(*layers)
      self.model = model_ft.to(self.device)  
      self.linear = torch.nn.Linear(256*6*6,4096).to(self.device)  

    elif choice == "VGG-19":
      self.model_name = "VGG-19"
      model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

      layers = list(model_ft.children())[:-1]
      # remove classifier layer
      model_ft = nn.Sequential(*layers)
      self.model = model_ft.to(self.device)
      self.linear = nn.Linear(512*7*7,4096).to('cuda') 

    

    # Freeze parameters so we don't backprop through them
    for param in self.model.parameters():
        param.requires_grad = False

    # Fine tune the last layers
    if fine_tune:
      for layer in list(self.model.children())[-2:]:
        for param in layer.parameters():
          param.requires_grad = True

  
  def forward(self, X):
    # think about the ensemble and dropout a bit more
    features = self.model(X)
    if self.linear:
      features_flatten = features.flatten()
      features = self.linear(features_flatten)
    return features


  def encode_image(self, image_path, normalize=False):
    """
    have to make this so that this can also process a lot of images at the same time
    """
    img = Image.open(image_path)
    img_processed = self.transform(img).unsqueeze(0)

    if torch.cuda.is_available():
      img_processed = img_processed.to('cuda')
      self.model.to('cuda')

    if not self.trained:
      print("Model has not been trained/finetuned")
    with torch.no_grad():
      output = self.model(img_processed)[0]

    if normalize:
      # add normalization code here
      output = torch.nn.functional.softmax(output, dim=0)

    return output
