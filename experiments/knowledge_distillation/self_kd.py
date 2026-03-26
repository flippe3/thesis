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

train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])

config = {
    'SAVE': True,
    'NAME': "online_resnet20",
    'TEACHER': "teacher/resnet110-cifar100.bin",
    'SAVE_DIR': "../../teacher_models/",
    'BATCH_SIZE': 64,
    'EPOCHS': 240,
    'LEARNING_RATE': 0.05,
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 5e-4,
    'METHOD:': "OFFLINE",
}

run = wandb.init(
#   mode="disabled",
  project="new_thesis",
#   dir="../../",
  name = config['NAME'],
  config=config,
  tags=["cifar100"]
)

train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

teacher = model_dict['resnet110'](num_classes=100)
teacher.to(device)

student = model_dict['resnet20'](num_classes=100)
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

def train(epochs, teacher_optimizer, student_optimizer, train_loader, test_loader, student_model, teacher_model):
    s_best_loss = float('inf')
    t_best_loss = float('inf')
    s_best_model = None
    t_best_model = None
    for epoch in range(epochs):
        student_model.train()
        teacher_model.train()
        for _, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.to(device)
            student_pred = student_model(data)
            teacher_pred = teacher_model(data)

            teacher_loss = loss_fn(teacher_pred, labels)
            student_loss = kd_loss(student_pred, teacher_pred, labels)

            teacher_loss.backward(retain_graph=True)
            student_loss.backward()

            teacher_optimizer.step()
            student_optimizer.step()

            teacher_optimizer.zero_grad()
            student_optimizer.zero_grad()
        
        student_model.eval()
        teacher_model.eval()
        s_loss, s_acc, s_f1 = validation(student_model, test_loader)
        print(f"STUDENT Epoch {epoch+1} - val_acc: {s_acc} - val_loss: {s_loss}")
        wandb.log({'val_f1': s_f1, 'val_acc': s_acc, 'val_loss': s_loss})
        

        t_loss, t_acc, t_f1 = validation(teacher_model, test_loader)
        print(f"Teacher Epoch {epoch+1} - val_acc: {t_acc} - val_loss: {t_loss}")
        wandb.log({'t_val_f1': t_f1, 't_val_acc': t_acc, 't_val_loss': t_loss})

        # Update learning rate at 150,180,210 epochs
        if epoch in [150, 180, 210]:
            for param_group in student_optimizer.param_groups:
                param_group['lr'] *= 0.1
            for param_group in teacher_optimizer.param_groups:
                param_group['lr'] *= 0.1        

        # Early stopping without the stopping :)
        if s_loss < s_best_loss:
            s_best_loss = s_loss
            s_best_model = copy.deepcopy(student_model.state_dict())

        # Early stopping without the stopping :)
        if t_loss < t_best_loss:
            t_best_loss = t_loss
            t_best_model = copy.deepcopy(teacher_model.state_dict())

    return t_best_model, s_best_model
        

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())
print(f"Teacher param count:{teacher_params}")
print(f"Student param count:{student_params} ({student_params/teacher_params})")

teacher_loss, teacher_acc, teacher_f1 = validation(teacher, test_loader)
print(f"Teacher Acc: {teacher_acc}")

loss_fn = torch.nn.CrossEntropyLoss()
student_optimizer = torch.optim.SGD(student.parameters(), lr=config['LEARNING_RATE'], momentum=config['MOMENTUM'], weight_decay=config['WEIGHT_DECAY'])
teacher_optimizer = torch.optim.SGD(teacher.parameters(), lr=config['LEARNING_RATE'], momentum=config['MOMENTUM'], weight_decay=config['WEIGHT_DECAY'])
t_best_model, s_best_model = train(config['EPOCHS'], teacher_optimizer, student_optimizer,  train_loader, val_loader, student, teacher)

test_loss, test_acc, test_f1 = validation(student, test_loader)
print(f"Last Acc: {test_acc} - Last Loss: {test_loss}")
wandb.log({'last_f1': test_acc, 'last_acc': test_acc, 'last_loss': test_loss})

# Load our early stopping model
student.load_state_dict(s_best_model)
student.eval()
teacher.load_state_dict(t_best_model)
teacher.eval()

test_loss, test_acc, test_f1 = validation(student, test_loader)
print(f"STUDENT Test Acc: {test_acc} - Test Loss: {test_loss}")
wandb.log({'s_test_f1': test_acc, 's_test_acc': test_acc, 's_test_loss': test_loss})

test_loss, test_acc, test_f1 = validation(teacher, test_loader)
print(f"TEACHER Test Acc: {test_acc} - Test Loss: {test_loss}")
wandb.log({'t_test_f1': test_acc, 't_test_acc': test_acc, 't_test_loss': test_loss})

if config['SAVE']:
    torch.save(student.state_dict(), config['SAVE_DIR'] + config['NAME'])
    torch.save(teacher.state_dict(), config['SAVE_DIR'] + "o_teacher" +  config['NAME'])
