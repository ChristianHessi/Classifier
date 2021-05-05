from torch import nn, optim
import torch
from torchvision import datasets, transforms, models

# Set the model and optimizer with the correct architecture
def load_correct_model(model, arch, hidden_layers, learning_rate=0.005):
    optimizer = None
    if arch in ["vgg16", "vgg19", "alexnet"]:
        classifier = nn.Sequential(nn.Linear(4096, hidden_layers),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(hidden_layers, 102),
                           nn.LogSoftmax(dim=1))
        model.classifier[6] = classifier
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=learning_rate)
    
    elif arch in ["resnet101", "resnet152", "resnet50"]:
        fc = nn.Sequential(nn.Linear(2048, hidden_layers),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(hidden_layers, 102),
                           nn.LogSoftmax(dim=1))
        model.fc = fc
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
        
    return model, optimizer

#Save the model with arch to buil easily later
def save_model(model, arch, hidden_layers, optimizer, save_dir):
    checkpoint = {
        "arch": arch,
        "hidden_layers" : hidden_layers,
        "class_to_idx" : model.class_to_idx,
        "state_dict" : model.state_dict(),
        "optimizer_dict" : optimizer.state_dict()
    }
    
    torch.save(checkpoint, save_dir + "checkpoint.pth")

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location= 'cuda:0' if device == 'cuda' else 'cpu')
    model = getattr(models, checkpoint["arch"])(pretrained = True)
    
    model, optimizer = load_correct_model(model, checkpoint["arch"], checkpoint["hidden_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model
    
    