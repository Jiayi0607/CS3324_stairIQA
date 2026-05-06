import os
import time
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import models.stairIQA_resnet as stairIQA_resnet

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Single Image Quality Assessment with Performance Analysis")
    parser.add_argument('--model_path', required=True, help='Path to model weights')
    parser.add_argument('--image_path', required=True, help='Path to test image')
    parser.add_argument('--test_method', default='five', choices=['one', 'five'],
                       help='Test method: one crop (center) or five crop')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs for throughput measurement')
    return parser.parse_args()

def load_model(model_path, device):
    """Load model from checkpoint"""
    model = stairIQA_resnet.resnet50(pretrained=False)
    
    # Remove DataParallel prefix if exists
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, test_method):
    """Preprocess image for inference"""
    img = Image.open(image_path).convert('RGB')
    
    if test_method == 'one':
        transform = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0)  # [1, 3, 320, 320]
    else:
        # Five crop returns 5 crops: shape will be [5, 3, 320, 320]
        transform = transforms.Compose([
            transforms.Resize(384),
            transforms.FiveCrop(320),
            lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
            lambda crops: torch.stack([transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])(crop) for crop in crops])
        ])
        crops = transform(img)  # [5, 3, 320, 320]
        return crops.unsqueeze(0)  # [1, 5, 3, 320, 320] for batch processing

def predict_single_image(model, input_tensor, test_method):
    """Predict quality score for single image with proper crop handling"""
    with torch.no_grad():
        if test_method == 'one':
            # Input shape: [1, 3, 320, 320]
            output = model(input_tensor)
            
            if isinstance(output, tuple):
                output = output[-1]
            
            score = output.item() if output.numel() == 1 else output.mean().item()
            return score
            
        else:
            # Five crop: input shape [1, 5, 3, 320, 320]
            bs, ncrops, c, h, w = input_tensor.size()
            
            # Process all crops at once
            input_reshaped = input_tensor.view(-1, c, h, w)  # [5, 3, 320, 320]
            outputs = model(input_reshaped)
            
            if isinstance(outputs, tuple):
                outputs = outputs[-1]
            
            # Reshape outputs back and average
            outputs = outputs.view(bs, ncrops, -1).mean(1)  # Average across crops
            score = outputs.item()
            return score

def measure_performance(model, input_tensor, test_method, num_runs=100):
    """Measure inference performance with proper crop handling"""
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            if test_method == 'one':
                _ = model(input_tensor)
            else:
                # Five crop: need to reshape
                bs, ncrops, c, h, w = input_tensor.size()
                input_reshaped = input_tensor.view(-1, c, h, w)
                _ = model(input_reshaped)
    
    # Measure latency
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            if test_method == 'one':
                _ = model(input_tensor)
            else:
                bs, ncrops, c, h, w = input_tensor.size()
                input_reshaped = input_tensor.view(-1, c, h, w)
                _ = model(input_reshaped)
        latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    return {
        'avg_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'throughput': 1000 / np.mean(latencies)  # images per second
    }

def analyze_model(model):
    """Analyze model complexity"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate FLOPs (approximate)
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, 320, 320).to(next(model.parameters()).device)
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = macs / 1e9
    except ImportError:
        macs, gflops = "N/A (install thop)", "N/A"
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'macs': macs,
        'gflops': gflops,
        'model_size_mb': total_params * 4 / (1024**2)
    }

def print_report(model_info, perf_info, prediction, test_method):
    """Print formatted analysis report"""
    print("\n" + "="*60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*60)
    
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")
    
    if isinstance(model_info['macs'], (int, float)):
        print(f"FLOPs (MACs): {model_info['macs'] / 1e9:.2f} G")
        print(f"GFLOPs: {model_info['gflops']:.2f}")
    else:
        print(f"FLOPs (MACs): {model_info['macs']}")
        print(f"GFLOPs: {model_info['gflops']}")
    
    print(f"Estimated model size (FP32): {model_info['model_size_mb']:.2f} MB")
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"Test Method: {test_method}")
    print(f"Latency Statistics (ms):")
    print(f"  Average: {perf_info['avg_latency']:.2f} ± {perf_info['std_latency']:.2f}")
    print(f"  Min: {perf_info['min_latency']:.2f}")
    print(f"  Max: {perf_info['max_latency']:.2f}")
    print(f"  Std Dev: {perf_info['std_latency']:.2f}")
    print(f"\nThroughput: {perf_info['throughput']:.2f} images/second")
    
    print("\n" + "="*60)
    print("SINGLE IMAGE PREDICTION")
    print("="*60)
    
    print(f"Image Quality Score: {prediction['score']:.6f}")
    print(f"Inference Time: {prediction['inference_time']:.2f} ms")

def main():
    args = parse_args()
    
    # Check files exist
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Image: {os.path.basename(args.image_path)}")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Test Method: {args.test_method}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Preprocess image
    input_tensor = preprocess_image(args.image_path, args.test_method).to(device)
    
    # Model analysis
    model_info = analyze_model(model)
    
    # Performance measurement
    perf_info = measure_performance(model, input_tensor, args.test_method, args.runs)
    
    # Single prediction (with timing)
    start_time = time.perf_counter()
    score = predict_single_image(model, input_tensor, args.test_method)
    inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
    prediction = {'score': score, 'inference_time': inference_time}
    
    # Print report
    print_report(model_info, perf_info, prediction, args.test_method)
    
    return score

if __name__ == '__main__':
    main()