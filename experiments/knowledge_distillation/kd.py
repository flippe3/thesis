import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch.nn.functional as F
import os
from models import model_dict
import copy

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

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

train_dataset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=test_transform)
train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])

config = {
    'SAVE': True,
    'NAME': "resnet20",
    'TEACHER': "teacher/resnet110-cifar10-baseline.pt",
    'SAVE_DIR': "../../teacher_models/",
    'BATCH_SIZE': 64,
    'EPOCHS': 240,
    'LEARNING_RATE': 0.05,
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 5e-4,
}

run = wandb.init(
  mode="disabled",
  project="meta-kd",
  dir="../../",
  name = config['NAME'],
  config=config,
  tags=["cifar10"]
)

train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

teacher = model_dict['resnet110'](num_classes=10)
teacher.load_state_dict(torch.load(config['TEACHER']))
teacher.to(device)
teacher.eval()

student = model_dict['resnet20'](num_classes=10)
student.to(device)

def validation(model, data_loader):
    all_labels, all_preds = [], []
    val_loss = 0
    with torch.no_grad():
        for _, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)

            pred = model(data)
            
            loss = F.cross_entropy(pred, labels)
            val_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().data.numpy())
        val_loss /= len(val_loader)
        return val_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro") 

def dist_loss(teacher, student, temperature=4):
    log_prob_student = F.log_softmax(student/temperature, dim=1)
    soft_targets = F.softmax(teacher/temperature, dim=1)
    return -(soft_targets * log_prob_student).sum(dim=1).mean()

def kd_loss(student_output, teacher_output, labels, alpha=0.9):
    soft_loss = (1 - alpha) * dist_loss(teacher_output, student_output)
    hard_loss = alpha * F.cross_entropy(student_output, labels)
    loss = soft_loss + hard_loss
    return loss

def train(epochs, optimizer, train_loader, test_loader, student_model, teacher_model):
    best_loss = float('inf')
    best_model = None
    for epoch in range(epochs):
        student_model.train()
        for _, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.to(device)
            student_pred = student_model(data)
            teacher_pred = teacher_model(data)
            loss = kd_loss(student_pred, teacher_pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        
        student_model.eval()
        loss, acc, f1 = validation(student_model, test_loader)
        print(f"Epoch {epoch+1} - val_acc: {acc} - val_loss: {loss}")
        wandb.log({'val_f1': f1, 'val_acc': acc, 'val_loss': loss})

        # Update learning rate at 150,180,210 epochs
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # Early stopping without the stopping :)
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(student_model.state_dict())

    return best_model
        

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())
print(f"Teacher param count:{teacher_params}")
print(f"Student param count:{student_params} ({student_params/teacher_params})")

teacher_loss, teacher_acc, teacher_f1 = validation(teacher, test_loader)
print(f"Teacher Acc: {teacher_acc}")

optimizer = torch.optim.SGD(student.parameters(), lr=config['LEARNING_RATE'], momentum=config['MOMENTUM'], weight_decay=config['WEIGHT_DECAY'])
best_model = train(config['EPOCHS'], optimizer, train_loader, val_loader, student, teacher)

test_loss, test_acc, test_f1 = validation(student, test_loader)
print(f"Last Acc: {test_acc} - Last Loss: {test_loss}")
wandb.log({'last_f1': test_acc, 'last_acc': test_acc, 'last_loss': test_loss})

# Load our early stopping model
student.load_state_dict(best_model)
student.eval()

test_loss, test_acc, test_f1 = validation(student, test_loader)
print(f"Test Acc: {test_acc} - Test Loss: {test_loss}")
wandb.log({'test_f1': test_acc, 'test_acc': test_acc, 'test_loss': test_loss})

if config['SAVE']:
    torch.save(student.state_dict(), config['SAVE_DIR'] + config['NAME'])