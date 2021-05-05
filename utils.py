import argparse
import json
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
#set function to read terminal command

def read_train_command():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers/', help="where to find images")
    parser.add_argument('--arch', type=str, default='vgg16', help="pretrained model to use (default = vgg16)")
    parser.add_argument('--hidden_units', type=int, default=256, help="hidden units for model (default=5 1 2)")
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="learning rate, (default = 0.0005)")
    parser.add_argument('--device', type=str, default='cuda', help="Choose between gpu and cpu, (default = cuda)")
    parser.add_argument('--epochs', type=int, default=3, help="set the epochs of your training model, (default = 3)")
    parser.add_argument('--save_dir', type=str, default='./', help="Directory to save the checkpoint, (default = './')")
    
    return parser.parse_args()

def read_predict_command():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type=str, default='flowers/', help="the path of the image to predict")
    parser.add_argument('checkpoint', type=str, default='vgg16', help="checkpoint of previous trained model")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="mapping of category to real names")
    parser.add_argument('--top_k', type=int, default=5, help="top k values")
    parser.add_argument('--gpu', default="cpu", help="activate gpu mode for inference")
    
    return parser.parse_args()

#check the inputs arguments from terminal
#ref: (https://github.com/stk0919/ImageClassifier/blob/main/train%2020210208.py)
def check_args_input(args):
    if(args.save_dir[0] in ["/", "\\"] and args.save_dir[-1] != "/" and len(args.save_dir) > 0 and os.exist(args.save_dir)):
        print("Enter your save directory without the preceding slash '/' but must end with slash")
        print("Example: ")
        print("  Checkpoint/      <- This is good")
        print("  /Checkpoint     <- this will not work")
        quit()
    
    if(args.data_dir[0] in ["/", "\\"] and args.save_dir[-1] != "/" and len(args.save_dir) > 0 and os.exist(args.save_dir)):
        print("Enter your save directory without the preceding slash '/' but must end with slash")
        print("Example: ")
        print("  flowers/      <- This is good")
        print("  /flowers     <- this will not work")
        quit()

    if(args.arch not in ["vgg16", "vgg19", "alexnet", "resnet101", "resnet152", "resnet50"]):
        print("Choose an available CNN network from the following:")    
        print("-  VGG16 (default)")
        print("-  VGG19")   
        print("-  resnet101")
        print("-  resnet152")
        print("-  resnet50")
        print("-  alexnet")
        quit()

    if(not(args.learning_rate > 0 and args.learning_rate < 1)):
        print("Please enter a valid learn rate between 0 and 1 (exclusive of 0 and 1)")
        quit()

    if(args.hidden_units <= 0):
        print("Please enter a valid hidden unit value greater than 0. Default is 1024")                    
        quit()          

    if(args.epochs <= 0):
        print("Please enter a valid number of epochs greater than 0. Default is 3")                    
        quit()    

    if args.device not in ['cpu', 'cuda']:
        print("Please enter 'cuda' or 'cpu' for the device")                    
        quit()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("cuda is not detected, please enable it or run in cpu mode") 
        quit()
    
    print("\nIt's all good here, GREAT")
def check_predict_command(args):
#     if(args.image_path != N)
    pass
        
        
# function to transform image to tensor
def image_transforms():
    im_transform = {
        'train' : transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
        
        'valid' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    }
    return im_transform

def get_label_mapping(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def get_datasets(data_dir):
    data = {
        'train' : datasets.ImageFolder(root=data_dir+'/train', transform=image_transforms()['train']),
        'valid' : datasets.ImageFolder(root=data_dir+'/valid', transform=image_transforms()['valid']),
    }
    
    return data

def get_data_loader(data_dir):
    data = get_datasets(data_dir)
    dataloader = {
        'train' : torch.utils.data.DataLoader(data['train'], batch_size=64, shuffle=True),
        'valid' : torch.utils.data.DataLoader(data['valid'], batch_size=64),
    }
    
    return dataloader
        