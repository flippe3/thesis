import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
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
    'NAME': 'student_cifar10.pth',
    'TEACHER': 'path/to/teacher_model.pth'  # Replace with your actual path
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
train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])

train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

# Define a standard CNN model
class StandardCNN(nn.Module):
    def __init__(self):
        super(StandardCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Adjust for CIFAR-10 input size
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load teacher model
teacher_model = model_dict['resnet110'](num_classes=10)
teacher_model.load_state_dict(torch.load(config['TEACHER']))
teacher_model.to(device)
teacher_model.eval()

# Initialize student model
student_model = StandardCNN().to(device)

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

# Knowledge distillation loss function
def kd_loss(student_output, teacher_output, labels, alpha=0.9):
    soft_loss = F.kl_div(F.log_softmax(student_output, dim=1), F.softmax(teacher_output, dim=1), reduction='batchmean')
    hard_loss = F.cross_entropy(student_output, labels)
    loss = (1 - alpha) * soft_loss + alpha * hard_loss
    return loss

# Training function
def train(epochs, optimizer, train_loader, val_loader, student_model, teacher_model):
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
        
        # Validation step
        student_model.eval()
        val_loss, acc, f1 = validation(student_model, val_loader)
        print(f"Epoch {epoch + 1} - val_acc: {acc:.4f} - val_loss: {val_loss:.4f}")

        # Learning rate decay
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(student_model.state_dict())

    return best_model

# Count parameters
teacher_params = sum(p.numel() for p in teacher_model.parameters())
student_params = sum(p.numel() for p in student_model.parameters())
print(f"Teacher param count: {teacher_params}")
print(f"Student param count: {student_params} ({student_params / teacher_params:.2f})")

# Validate teacher model
teacher_loss, teacher_acc, teacher_f1 = validation(teacher_model, test_loader)
print(f"Teacher Acc: {teacher_acc:.4f}")

# Optimizer for student model
optimizer = torch.optim.SGD(student_model.parameters(), lr=config['LEARNING_RATE'], momentum=config['MOMENTUM'], weight_decay=config['WEIGHT_DECAY'])

# Train student model
best_model = train(config['EPOCHS'], optimizer, train_loader, val_loader, student_model, teacher_model)

# Load best model
student_model.load_state_dict(best_model)
student_model.eval()

# Validate student model on test set
test_loss, test_acc, test_f1 = validation(student_model, test_loader)
print(f"Test Acc: {test_acc:.4f} - Test Loss: {test_loss:.4f}")

# Save student model
if config['SAVE']:
    os.makedirs(config['SAVE_DIR'], exist_ok=True)  # Ensure the save directory exists
    torch.save(student_model.state_dict(), os.path.join(config['SAVE_DIR'], config['NAME']))
