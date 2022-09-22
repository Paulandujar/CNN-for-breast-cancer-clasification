from torchvision import models, transforms, datasets
import torch
import os
import torch.nn.functional as F
import cv2
from patchify import patchify
import numpy as np
import shutil
from PIL import Image
import torch.optim as optim
import torch.nn as nn


def test(folder, image_path):

  # Se procesa la imagen  y se divide en patches
  image = cv2.imread(image_path, 1)
  patches_data = patchify(image, (224,224,3), step=112)

  patches = []
  cnt = 0
  for x in patches_data:
    cnt += 1
    for y in patches_data[cnt-1]:

      patches.append(y[0])

  # Guardado de patches en directorio
  n = 0
  for im in patches:
    n += 1
    data = Image.fromarray(im)
    imagepath = str('data/patches-test/test/pat-' + str(n) + '.png')
    data.save(imagepath)

  # Se carga el modelo (resnet18)
  model_path = "data/resnet20epoch0.001LR-state_dict-new.pth"

  model = models.resnet18()
  model.load_state_dict(torch.load(model_path, 'cpu'))

  device =  "cpu"
  model = model.to(device)

  # Creación dataloader
  data_transforms = {
      'test': transforms.Compose([
          transforms.ToTensor()
      ])}
    
  image_dataset = datasets.ImageFolder(folder, data_transforms['test'])
  dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

  # Ejecutamos test
  model.eval()
  cnt = 0
  patches_preds = []
  
  with torch.no_grad():
    for inputs, classes in dataloader:
      cnt += 1
      inputs = inputs.to(device)

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      predicted = predicted.cpu().numpy()


      for i in predicted:
        patches_preds.append(i)
      
    numPatchesMalignos = sum(patches_preds)
    numPatches = len(patches_preds)
    umbral = 0.90
    
    if numPatchesMalignos/numPatches > umbral:
      resultado = 'Positivo en cáncer'
    else:
      resultado = 'Negativo en cáncer' 

    return(resultado)



