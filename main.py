import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import torch.nn as nn
import torch.optim as optim
import copy
from config import *
from cnn import SimpleCNN
from helpers import train_one_epoch, evaluate, print_size_of_model, quantize_to_float8
from visualize import plot_performance_comparison, visualize_predictions, plot_quantization_impact

if __name__ == "__main__":
    print("--- 1. Training Baseline Float32 Model ---")
    float_model = SimpleCNN().to(TRAIN_DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(float_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_one_epoch(float_model, trainloader, optimizer, criterion, TRAIN_DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    float_model.to(QUANT_DEVICE)
    float_model.eval()
    
    print("\n--- 2. Evaluating Baseline ---")
    acc_fp32, time_fp32 = evaluate(float_model, testloader, QUANT_DEVICE)
    size_fp32 = print_size_of_model(float_model, "Float32")
    print(f"Float32 Accuracy: {acc_fp32:.2f}%")
    print(f"Float32 Inference Time: {time_fp32:.4f} ms/sample")

    print("\n--- 3. Performing Static Quantization ---")
    quantized_model = copy.deepcopy(float_model)
    quantized_model.eval()
    
    torch.backends.quantized.engine = 'qnnpack'
    
    quantized_model.fuse_model()
    
    quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    
    torch.ao.quantization.prepare(quantized_model, inplace=True)
    
    print("Calibrating (feeding data to observers)...")
    evaluate(quantized_model, testloader, QUANT_DEVICE) 
    
    torch.ao.quantization.convert(quantized_model, inplace=True)
    
    print("\n--- 4. Evaluating Int8 Quantized Model ---")
    acc_int8, time_int8 = evaluate(quantized_model, testloader, QUANT_DEVICE)
    size_int8 = print_size_of_model(quantized_model, "Int8")
    print(f"Int8 Accuracy: {acc_int8:.2f}%")
    print(f"Int8 Inference Time: {time_int8:.4f} ms/sample")

    print("\n--- 5. Performing Float8 Quantization ---")
    float8_model, size_float8 = quantize_to_float8(float_model)
    
    print("\n--- 6. Evaluating Float8 Quantized Model ---")
    acc_float8, time_float8 = evaluate(float8_model, testloader, QUANT_DEVICE)
    print(f"Float8 Size: {size_float8:.2f} MB (8-bit quantized weights)")
    print(f"Float8 Accuracy: {acc_float8:.2f}%")
    print(f"Float8 Inference Time: {time_float8:.4f} ms/sample")

    speedup_int8 = time_fp32 / time_int8
    speedup_float8 = time_fp32 / time_float8
    
    print("\n" + "="*75)
    print("FINAL RESULTS TABLE")
    print("="*75)
    print(f"{'Metric':<22} | {'Float32':<13} | {'Int8':<13} | {'Float8':<13}")
    print("-" * 75)
    print(f"{'Size (MB)':<22} | {size_fp32:<13.2f} | {size_int8:<13.2f} | {size_float8:<13.2f}")
    print(f"{'Accuracy (%)':<22} | {acc_fp32:<13.2f} | {acc_int8:<13.2f} | {acc_float8:<13.2f}")
    print(f"{'Inference Latency (ms)':<22} | {time_fp32:<13.4f} | {time_int8:<13.4f} | {time_float8:<13.4f}")
    print("-" * 75)
    print(f"{'Size Reduction':<22} | {'1.00x':<13} | {size_fp32/size_int8:<.2f}x{' '*8} | {size_fp32/size_float8:<.2f}x")
    print(f"{'Memory Savings':<22} | {'0.0%':<13} | {(1-size_int8/size_fp32)*100:<.1f}%{' '*8} | {(1-size_float8/size_fp32)*100:<.1f}%")
    print(f"{'Inference Speedup':<22} | {'1.00x':<13} | {speedup_int8:<.2f}x{' '*8} | {speedup_float8:<.2f}x")
    print("="*75)

    print("\n--- 7. Generating Visualizations ---")
    plot_performance_comparison(acc_fp32, acc_int8, acc_float8, time_fp32, time_int8, time_float8, size_fp32, size_int8, size_float8)
    visualize_predictions(float_model, quantized_model, float8_model, num_samples=12)
    plot_quantization_impact(acc_fp32, acc_int8, acc_float8, time_fp32, time_int8, time_float8, size_fp32, size_int8, size_float8)
    print("Visualizations saved: performance_comparison.png, fmnist_predictions.png, quantization_impact.png")
