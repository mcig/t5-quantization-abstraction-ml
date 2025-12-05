import matplotlib.pyplot as plt
import numpy as np
import torch
from config import testset, testloader, QUANT_DEVICE

FASHION_MNIST_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_performance_comparison(acc_fp32, acc_int8, acc_float8, time_fp32, time_int8, time_float8, size_fp32, size_int8, size_float8):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = ['Float32', 'Int8', 'Float8']
    accuracies = [acc_fp32, acc_int8, acc_float8]
    times = [time_fp32, time_int8, time_float8]
    sizes = [size_fp32, size_int8, size_float8]
    
    axes[0].bar(models, accuracies, color=['#3498db', '#e74c3c', '#9b59b6'])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylim([0, 115])  # Increased limit to fit labels
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    axes[1].bar(models, times, color=['#3498db', '#e74c3c', '#9b59b6'])
    axes[1].set_ylabel('Latency (ms/sample)')
    axes[1].set_title('Inference Latency Comparison')
    axes[1].set_ylim([0, max(times) * 1.2])  # Dynamic limit
    for i, v in enumerate(times):
        axes[1].text(i, v + max(times)*0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    axes[2].bar(models, sizes, color=['#3498db', '#e74c3c', '#9b59b6'])
    axes[2].set_ylabel('Model Size (MB)')
    axes[2].set_title('Model Size Comparison')
    axes[2].set_ylim([0, max(sizes) * 1.2])  # Dynamic limit
    for i, v in enumerate(sizes):
        axes[2].text(i, v + max(sizes)*0.02, f'{v:.2f}MB', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./visuals/performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def visualize_predictions(float_model, quantized_model, float8_model, num_samples=12):
    float_model.eval()
    quantized_model.eval()
    float8_model.eval()
    
    sample_images = []
    sample_labels = []
    float_preds = []
    int8_preds = []
    float8_preds = []
    float_probs = []
    int8_probs = []
    float8_probs = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(QUANT_DEVICE)
            labels = labels.to(QUANT_DEVICE)
            
            float_outputs = float_model(images)
            int8_outputs = quantized_model(images)
            float8_outputs = float8_model(images)
            
            float_probs_batch = torch.nn.functional.softmax(float_outputs, dim=1)
            int8_probs_batch = torch.nn.functional.softmax(int8_outputs, dim=1)
            float8_probs_batch = torch.nn.functional.softmax(float8_outputs, dim=1)
            
            float_pred_batch = torch.argmax(float_outputs, dim=1)
            int8_pred_batch = torch.argmax(int8_outputs, dim=1)
            float8_pred_batch = torch.argmax(float8_outputs, dim=1)
            
            for i in range(min(num_samples, len(images))):
                img = images[i].cpu().squeeze()
                img = img * 0.5 + 0.5
                img = torch.clamp(img, 0, 1)
                sample_images.append(img.numpy())
                sample_labels.append(labels[i].item())
                float_preds.append(float_pred_batch[i].item())
                int8_preds.append(int8_pred_batch[i].item())
                float8_preds.append(float8_pred_batch[i].item())
                float_probs.append(float_probs_batch[i].cpu().numpy())
                int8_probs.append(int8_probs_batch[i].cpu().numpy())
                float8_probs.append(float8_probs_batch[i].cpu().numpy())
            
            if len(sample_images) >= num_samples:
                break
    
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples*1.5, 8))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_samples):
        img = sample_images[idx]
        true_label = sample_labels[idx]
        float_pred = float_preds[idx]
        int8_pred = int8_preds[idx]
        float8_pred = float8_preds[idx]
        
        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].set_title(f'True: {FASHION_MNIST_LABELS[true_label]}', fontsize=8)
        axes[0, idx].axis('off')
        
        float_conf = float_probs[idx][float_pred] * 100
        int8_conf = int8_probs[idx][int8_pred] * 100
        float8_conf = float8_probs[idx][float8_pred] * 100
        
        color_fp32 = 'green' if float_pred == true_label else 'red'
        color_int8 = 'green' if int8_pred == true_label else 'red'
        color_float8 = 'green' if float8_pred == true_label else 'red'
        
        axes[1, idx].barh([0], [float_conf], color=color_fp32, alpha=0.7)
        axes[1, idx].set_xlim([0, 100])
        axes[1, idx].set_title(f'FP32: {FASHION_MNIST_LABELS[float_pred]}\n({float_conf:.1f}%)', fontsize=7)
        axes[1, idx].set_yticks([])
        if idx == 0: axes[1, idx].set_ylabel('Confidence', fontsize=8)
        
        axes[2, idx].barh([0], [int8_conf], color=color_int8, alpha=0.7)
        axes[2, idx].set_xlim([0, 100])
        axes[2, idx].set_title(f'Int8: {FASHION_MNIST_LABELS[int8_pred]}\n({int8_conf:.1f}%)', fontsize=7)
        axes[2, idx].set_yticks([])
        if idx == 0: axes[2, idx].set_ylabel('Confidence', fontsize=8)
        
        axes[3, idx].barh([0], [float8_conf], color=color_float8, alpha=0.7)
        axes[3, idx].set_xlim([0, 100])
        axes[3, idx].set_title(f'Float8: {FASHION_MNIST_LABELS[float8_pred]}\n({float8_conf:.1f}%)', fontsize=7)
        axes[3, idx].set_yticks([])
        if idx == 0: axes[3, idx].set_ylabel('Confidence', fontsize=8)
        
        # Only show x-axis labels on the bottom row to reduce clutter
        axes[1, idx].set_xticks([])
        axes[2, idx].set_xticks([])
        axes[3, idx].tick_params(axis='x', labelsize=7)
    
    plt.suptitle('FashionMNIST Predictions: Float32 vs Int8 vs Float8 Quantized', fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig('./visuals/fmnist_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_quantization_impact(acc_fp32, acc_int8, acc_float8, time_fp32, time_int8, time_float8, size_fp32, size_int8, size_float8):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(3)
    width = 0.25
    
    axes[0, 0].bar(x - width, [acc_fp32, 0, 0], width, label='Float32', color='#3498db')
    axes[0, 0].bar(x, [0, acc_int8, 0], width, label='Int8', color='#e74c3c')
    axes[0, 0].bar(x + width, [0, 0, acc_float8], width, label='Float8', color='#9b59b6')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(['Float32', 'Int8', 'Float8'])
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 115])  # Increased limit
    for i, (val, offset) in enumerate([(acc_fp32, -width), (acc_int8, 0), (acc_float8, width)]):
        axes[0, 0].text(i + offset, val + 2, f'{val:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    axes[0, 1].bar(x - width, [time_fp32, 0, 0], width, label='Float32', color='#3498db')
    axes[0, 1].bar(x, [0, time_int8, 0], width, label='Int8', color='#e74c3c')
    axes[0, 1].bar(x + width, [0, 0, time_float8], width, label='Float8', color='#9b59b6')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].set_title('Latency Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Float32', 'Int8', 'Float8'])
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, max(time_fp32, time_int8, time_float8) * 1.25])  # Increased limit
    for i, (val, offset) in enumerate([(time_fp32, -width), (time_int8, 0), (time_float8, width)]):
        axes[0, 1].text(i + offset, val + max(time_fp32, time_int8, time_float8)*0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    axes[1, 0].bar(x - width, [size_fp32, 0, 0], width, label='Float32', color='#3498db')
    axes[1, 0].bar(x, [0, size_int8, 0], width, label='Int8', color='#e74c3c')
    axes[1, 0].bar(x + width, [0, 0, size_float8], width, label='Float8', color='#9b59b6')
    axes[1, 0].set_ylabel('Size (MB)')
    axes[1, 0].set_title('Size Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Float32', 'Int8', 'Float8'])
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, max(size_fp32, size_int8, size_float8) * 1.25])  # Increased limit
    for i, (val, offset) in enumerate([(size_fp32, -width), (size_int8, 0), (size_float8, width)]):
        axes[1, 0].text(i + offset, val + max(size_fp32, size_int8, size_float8)*0.02, f'{val:.2f}MB', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    size_reduction_int8 = (1 - size_int8/size_fp32) * 100
    size_reduction_float8 = (1 - size_float8/size_fp32) * 100
    speedup_int8 = time_fp32 / time_int8
    speedup_float8 = time_fp32 / time_float8
    acc_change_int8 = acc_int8 - acc_fp32
    acc_change_float8 = acc_float8 - acc_fp32
    
    metrics = ['Size Reduction\n(Int8)', 'Size Reduction\n(Float8)', 'Speedup\n(Int8)', 'Speedup\n(Float8)', 'Acc Change\n(Int8)', 'Acc Change\n(Float8)']
    values = [size_reduction_int8, size_reduction_float8, speedup_int8, speedup_float8, acc_change_int8, acc_change_float8]
    colors = ['#2ecc71', '#2ecc71', '#f39c12', '#f39c12', '#9b59b6', '#9b59b6']
    
    bars = axes[1, 1].bar(range(len(metrics)), values, color=colors)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Quantization Impact Metrics')
    axes[1, 1].set_xticks(range(len(metrics)))
    axes[1, 1].set_xticklabels(metrics, rotation=45, ha='right', fontsize=8)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Increase y-limit for impact metrics
    max_val = max([abs(v) for v in values])
    axes[1, 1].set_ylim([-max_val * 1.3, max_val * 1.3])
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        label = f'{val:.2f}%' if i < 2 or i >= 4 else f'{val:.2f}x'
        # Improved label positioning logic
        y_pos = val + (max_val * 0.1 if val >= 0 else -max_val * 0.15)
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, y_pos, 
                       label, ha='center', va='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./visuals/quantization_impact.png', dpi=150, bbox_inches='tight')
    plt.close()

