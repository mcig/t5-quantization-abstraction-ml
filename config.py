import torch
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
QUANT_DEVICE = torch.device("cpu")

print(f"Training on: {TRAIN_DEVICE}")
print(f"Quantizing on: {QUANT_DEVICE}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

