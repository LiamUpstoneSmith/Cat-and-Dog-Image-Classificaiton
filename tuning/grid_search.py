import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from itertools import product
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_prob=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(64*4*4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(accuracy=correct/total)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and transforms (replace with your dataset & dirs)
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

    # Replace 'YourDataset' below with your actual dataset
    full_dataset = datasets.ImageFolder(root='../-dataset/data', transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    learning_rates = [0.001, 0.0005]
    batch_sizes = [32, 64]
    dropout_probs = [0.3, 0.5]

    results = []

    for lr, batch_size, dropout in product(learning_rates, batch_sizes, dropout_probs):
        print(f"\nTraining with lr={lr}, batch_size={batch_size}, dropout={dropout}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = CNN(in_channels=3, num_classes=2, dropout_prob=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        epochs = 3
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            print(f"Train Loss: {train_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout': dropout,
            'val_accuracy': val_acc
        })

    df = pd.DataFrame(results)
    df.to_csv("grid_search_results.csv", index=False)
    print("\nGrid search complete. Results saved to grid_search_results.csv")

if __name__ == "__main__":
    main()

# Results:
# Lr,   Batch_size,  Dropout,  Val_Accuracy
# 0.001,  64,         0.3,      0.8013602720544108