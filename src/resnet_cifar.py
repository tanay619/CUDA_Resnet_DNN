import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse

# Hyperparameters
EPOCH = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Download CIFAR-10 dataset
image_dataset_downloader = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# Define CIFAR-10 Dataset class
class ImageDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing CIFAR10 data
        ])
        if self.datasplit == "train":
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=self.transform)
        else:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                        transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Define ResNet model
class Resnet_Q1(nn.Module):
    def __init__(self):
        super(Resnet_Q1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, 10)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# Trainer function
def trainer(gpu, dataloader, network, criterion, optimizer):
    device = torch.device("cuda:0" if gpu == "T" else "cpu")
    network = network.to(device)
    network.train()

    for epoch in range(EPOCH):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print("Training Epoch: {}, Loss: {:.3f}, Accuracy: {:.2f}%".format(epoch + 1, running_loss, accuracy))

    # Save the model checkpoint
    torch.save({'model_state_dict': network.state_dict()}, 'checkpoint.pth')


# Validator function
def validator(gpu, dataloader, network, criterion):
    device = torch.device("cuda:0" if gpu == "T" else "cpu")
    network = network.to(device)
    network.eval()

    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print("Validation: Loss: {:.3f}, Accuracy: {:.2f}%".format(running_loss, accuracy))


# Evaluator function
def evaluator(gpu, dataloader, network):
    device = torch.device("cuda:0" if gpu == "T" else "cpu")
    network = network.to(device)
    network.eval()

    checkpoint = torch.load('checkpoint.pth', map_location=device)
    network.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print("Evaluation: Loss: {:.3f}, Accuracy: {:.2f}%".format(running_loss, accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=str)
    args = parser.parse_args()

    imageDataset = {
        "train": ImageDataset(split="train"),
        "val": ImageDataset(split="val"),
        "test": ImageDataset(split="test")
    }

    network = Resnet_Q1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=network.parameters(), lr=LEARNING_RATE)

    for split in ["train", "val", "test"]:
        dataset = imageDataset[split]
        dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

        if split == "train":
            print(f"Training {network.__class__.__name__} on {split} split of CIFAR-10")
            trainer(args.gpu, dataloader, network, criterion, optimizer)
        elif split == "val":
            print(f"Validating {network.__class__.__name__} on {split} split of CIFAR-10")
            validator(args.gpu, dataloader, network, criterion)
        elif split == "test":
            print(f"Testing {network.__class__.__name__} on {split} split of CIFAR-10")
            evaluator(args.gpu, dataloader, network)
