import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_prob=0.3):
        """
        Improved CNN with BatchNorm, Dropout, and additional Conv layer.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            num_classes (int): Number of output classes.
            dropout_prob (float): Dropout probability before the final layer.
        """
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_prob)

        # Use adaptive average pooling to output a fixed size feature map regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layer input features = 64 channels * 4 * 4 spatial size
        self.fc1 = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.adaptive_pool(x)  # output size: batch x 64 x 4 x 4
        x = x.view(x.size(0), -1)  # flatten

        x = self.dropout(x)
        x = self.fc1(x)

        return x
