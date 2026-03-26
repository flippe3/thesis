import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import copy
import wandb
from models import model_dict

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.manual_seed(42)

config = {
    'NAME': 'resnet110-imagenet',
    'SAVE': True,
    'SAVE_DIR': "./",
    'BATCH_SIZE': 64,
    'EPOCHS': 240,
    'LEARNING_RATE': 0.05,
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 5e-4,
}

run = wandb.init(
#   mode="disabled",
  project="thesis",
  dir="../../wandb",
  name = config['NAME'],
  config=config,
  tags=["imagenet"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

model = model_dict['resnet110'](num_classes=1000)

model.to(device)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_dataset = torchvision.datasets.ImageNet(root='data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.ImageNet(root='data', train=False, download=True, transform=test_transform)
train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])

train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

def validation(model, data_loader):
    all_labels, all_preds = [], []
    val_loss = 0
    with torch.no_grad():
        for _, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)

            pred = model(data)
            
            loss = loss_fn(pred, labels)
            val_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().data.numpy())
        val_loss /= len(val_loader)
        return val_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro") 

def train(epochs, optimizer, train_loader, val_loader, model):
    best_loss = float('inf')
    best_model = None
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        all_train_preds, all_train_labels = [], []

        for _, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, labels)
            train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            loss.backward()
            optimizer.step()


        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")

        # Validation
        model.eval()
        val_loss, acc, f1 = validation(model, val_loader)
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
                   'val_loss': val_loss, 'val_acc': acc, 'val_f1': f1})
        print(f"Epoch {epoch+1} - Val Loss {val_loss}, F1: {f1} - Acc: {acc}")

        # Update learning rate at 150,180,210 epochs
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        # Early stopping without the stopping :)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

    return best_model


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'],momentum=config['MOMENTUM'])

best_model = train(config['EPOCHS'], optimizer, train_loader, val_loader, model)

test_loss, test_acc, test_f1 = validation(model, test_loader)
print(f"Last loss: {test_loss} - Last Acc: {test_acc} - Last F1: {test_f1}")
wandb.log({'last_loss': test_loss, 'last_acc': test_acc, 'last_f1': test_f1})

# Load our early stopping model
model.load_state_dict(best_model)
model.eval()

test_loss, test_acc, test_f1 = validation(model, test_loader)
print(f"Test loss: {test_loss} - Test Acc: {test_acc} - Test F1: {test_f1}")
wandb.log({'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1})

if config['SAVE']:
    torch.save(model.state_dict(), config['SAVE_DIR'] + config['NAME'] + ".bin")