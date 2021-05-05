
# Build a train model to classify flowers images corrctly

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import time
from workspace_utils import active_session
from PIL import Image
import random
import os
import numpy as np
from utils import *
from model_utils import *

def main():
    print("\n\nThe purpose of this application is to classify correctly any flowers images bases on Neural network training model\n\n"
          "to perform this classification you have to pass :\n\n"
          "\t- the data directory like : --data_dir [name_of_directory]. (default: flowers) ex: --data_dir flowers/ \n"
          "\t- the model architecture, choose between alexnet, vgg16, vgg19, resnet152, resnet50, resnet101 (default: vgg16). ex: --arch vgg16 \n"
          "\t- the hidden_units ex: --hidden_units 256\n"
          "\t- the learning_rate (between 0-1) ex: --learning_rate 0.1. default:256\n"
          "\t- the device (cpu or cuda) ex: --device cuda. default: cuda\n"
          "\t- the save directory to save checkpoint ex: --save_dir opt/. default: ''\n"
          "\t- the epochs ex: --epochs 20 default: 3\n\n")
    
    args = read_train_command()
    
    check_args_input(args)
    
    model = getattr(models, args.arch)(pretrained=True)
    device = args.device
    arch = args.arch
    epochs = args.epochs
    data_dir = args.data_dir
    save_dir = args.save_dir
    hu = args.hidden_units
    lr = args.learning_rate
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    model, optimizer = load_correct_model(model, arch, hu, lr)
    
    criterion = nn.NLLLoss()
    
    model.to(device)
    
    training_loss = 0
    
    with active_session():
        for e in range(epochs):
            for i, (images, labels) in enumerate(get_data_loader(data_dir)['train']):

                images, labels = images.to(device), labels.to(device)
                
                images.requires_grad = True

                optimizer.zero_grad()
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()

                optimizer.step()
                
                training_loss += loss.item()
                

            else:
                print("epoch: {}/{}".format(e+1, epochs))
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    # Extract imgages an names index from the valid loader 
                    for i, (images, labels) in enumerate(get_data_loader(data_dir)['valid']):
                        # Extract labels indices from cat_to_name

                        images, labels = images.to(device), labels.to(device)

                        log_ps = model(images)
                        loss = criterion(log_ps, labels)


                        valid_loss += loss

                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
    #                     print(labels, top_class, equals)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f"Training loss : {training_loss/((i+1)*len(get_data_loader(data_dir)['train'])):.3f}..."
                      f"Testing loss : {valid_loss/((i+1)*len(get_data_loader(data_dir)['valid'])):.3f}... "
                      f"Accuracy : {accuracy/(len(get_data_loader(data_dir)['valid'])):.3f}... "
                      "device : {}".format(device))
    
    model.class_to_idx = get_datasets(data_dir)['train'].class_to_idx
    save_model(model, arch, hu, optimizer, save_dir)
    
if __name__ == "__main__":
    main()
    