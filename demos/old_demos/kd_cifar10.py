import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import os
import copy

# Set up environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Configuration parameters
config = {
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 0.01,
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 5e-4,
    'EPOCHS': 200,
    'SAVE': True,
    'SAVE_DIR': './models/',
    'NAME_TEACHER': 'teacher_cifar10.pth',
    'NAME_STUDENT': 'student_cifar10.pth'
}

# Data transformations
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

# Define a ResNet model for the teacher
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Initialize the teacher model
teacher_model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)

# Validation function
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
            
    val_loss /= len(data_loader)
    return val_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")

# Training function for teacher model
def train(epochs, optimizer, train_loader, test_loader, model):
    best_loss = float('inf')
    best_model = None
    for epoch in range(epochs):
        model.train()
        for _, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        val_loss, acc, f1 = validation(model, test_loader)
        print(f"Epoch {epoch + 1} - val_acc: {acc:.4f} - val_loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

    return best_model

# Optimizer for teacher model
optimizer = torch.optim.SGD(teacher_model.parameters(), lr=config['LEARNING_RATE'], momentum=config['MOMENTUM'], weight_decay=config['WEIGHT_DECAY'])

# Train the teacher model
best_model = train(config['EPOCHS'], optimizer, train_loader, test_loader, teacher_model)

# Load best model
teacher_model.load_state_dict(best_model)
teacher_model.eval()

# Validate teacher model on test set
test_loss, test_acc, test_f1 = validation(teacher_model, test_loader)
print(f"Test Acc: {test_acc:.4f} - Test Loss: {test_loss:.4f}")

# Save teacher model
if config['SAVE']:
    os.makedirs(config['SAVE_DIR'], exist_ok=True)  # Ensure the save directory exists
    torch.save(teacher_model.state_dict(), os.path.join(config['SAVE_DIR'], config['NAME_TEACHER']))

# Define a simple CNN model for the student
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the student model
student_model = SimpleCNN().to(device)

# Optimizer for student model
optimizer_student = torch.optim.SGD(student_model.parameters(), lr=config['LEARNING_RATE'], momentum=config['MOMENTUM'], weight_decay=config['WEIGHT_DECAY'])

# Knowledge Distillation Loss
def distillation_loss(y_student, y_teacher, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y_student / T, dim=1), F.softmax(y_teacher / T, dim=1)) * (T * T) * alpha + F.cross_entropy(y_student, y_teacher.argmax(dim=1)) * (1. - alpha)

# Training function for student model with knowledge distillation
def train_student(epochs, optimizer, train_loader, test_loader, teacher_model, student_model, T, alpha):
    best_loss = float('inf')
    best_model = None
    for epoch in range(epochs):
        student_model.train()
        for _, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Get teacher model predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(data)

            student_outputs = student_model(data)
            loss = distillation_loss(student_outputs, teacher_outputs, T, alpha)
            loss.backward()
            optimizer.step()
        
        # Validation step
        student_model.eval()
        val_loss, acc, f1 = validation(student_model, test_loader)
        print(f"Epoch {epoch + 1} - val_acc: {acc:.4f} - val_loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(student_model.state_dict())

    return best_model

# Train the student model
T = 4.0  # Temperature for distillation
alpha = 0.7  # Weight for distillation loss
best_student_model = train_student(config['EPOCHS'], optimizer_student, train_loader, test_loader, teacher_model, student_model, T, alpha)

# Load best student model
student_model.load_state_dict(best_student_model)
student_model.eval()

# Validate student model on test set
test_loss_student, test_acc_student, test_f1_student = validation(student_model, test_loader)
print(f"Test Acc (Student): {test_acc_student:.4f} - Test Loss (Student): {test_loss_student:.4f}")

# Save student model
if config['SAVE']:
    os.makedirs(config['SAVE_DIR'], exist_ok=True)  # Ensure the save directory exists
    torch.save(student_model.state_dict(), os.path.join(config['SAVE_DIR'], config['NAME_STUDENT']))
