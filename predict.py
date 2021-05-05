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
    args = read_predict_command()
    check_predict_command(args)
    
    image_path = args.image_path
    checkpoint = args.checkpoint
    cat_to_name = get_label_mapping(args.category_names)
    topk = args.top_k
    device = args.gpu
    
    # Load the checkpoint
    model = load_model(checkpoint, device)
    
    # Predict image class
    process_image = image_transforms()['valid']
    
    # Processing the Image
    image = process_image(Image.open(image_path))
    image = image.type(torch.FloatTensor)
    image = image.unsqueeze(0)
    print(device)
#     model.to(device)
    model.eval()
    with torch.no_grad():
        # Launch prediction model to get probabilities
        log_ps = model(image)
        ps = torch.exp(log_ps)
        
        
        #get topk values
        top_ps, top_class = ps.topk(topk, dim=1)
        
        top_ps, top_class = np.array(top_ps)[0], np.array(top_class)[0]

        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class]
        class_names = [cat_to_name[str(e)] for e in top_class]
        
        print("The image is a {} and the predicted probability is {} \n the topk classes and probabilities is :\n {}, {}".format(class_names[0], top_ps[0], class_names, top_ps))
    
    
    
if __name__ == "__main__":
    main()