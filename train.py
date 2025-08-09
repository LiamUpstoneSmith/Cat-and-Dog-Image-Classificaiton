import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Progress bar
from dataset.dataset import create_dataset, CatsDogsDataset
from model.cnn import CNN

# Paths
cat_dir = "dataset/data/Cat"
dog_dir = "dataset/data/Dog"

# Load file paths & labels
x_train_files, x_test_files, y_train, y_test = create_dataset(cat_dir, dog_dir)

# Create Dataset objects
train_dataset = CatsDogsDataset(x_train_files, y_train)
test_dataset = CatsDogsDataset(x_test_files, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = CNN(in_channels=3, num_classes=2).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # tqdm progress bar
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)

        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

def evaluate_model():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    train_model()
    evaluate_model()

# Epoch 10 Average Loss: 0.3337
# Model saved to model.pth
# Test Accuracy: 83.72%