import torch
import time
import os
import copy
from config import testset

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    print(f"{label} Size: {size:.2f} MB")
    os.remove("temp.p")
    return size


def evaluate(model, loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    end_time = time.time()
    accuracy = 100 * correct / total
    inference_time = (end_time - start_time) * 1000 / len(testset)
    
    return accuracy, inference_time

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def quantize_to_float8(model):
    model_fp8 = copy.deepcopy(model)
    model_fp8.eval()
    
    total_fp8_size = 0
    with torch.no_grad():
        for name, module in model_fp8.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_fp32 = module.weight.data.clone()
                    try:
                        weight_fp8 = weight_fp32.to(torch.float8_e4m3fn)
                        weight_fp32_restored = weight_fp8.float()
                        module.weight.data = weight_fp32_restored
                        total_fp8_size += weight_fp8.numel()
                    except:
                        weight_fp8 = weight_fp32.to(torch.float8_e5m2)
                        weight_fp32_restored = weight_fp8.float()
                        module.weight.data = weight_fp32_restored
                        total_fp8_size += weight_fp8.numel()
    
    size_mb = (total_fp8_size * 1) / 1e6
    return model_fp8, size_mb

