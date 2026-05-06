import os
import argparse
import numpy as np
import torch
import json
import scipy.stats as stats
from torchvision import transforms
import models.stairIQA_resnet as stairIQA_resnet
from PIL import Image
from tqdm import tqdm  # Display progress bar

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Image Quality Assessment for StairIQA model (No DataParallel Version)")
    parser.add_argument('--model_path', help='Path to model weights file', required=True, type=str)
    parser.add_argument('--test_datasets', nargs='+', required=True, 
                        choices=['spaq_test', 'kadid_test', 'agiqa_test'],
                        help='List of test datasets to evaluate')
    parser.add_argument('--dataset_root', required=True, type=str,
                        help='Root directory of test datasets')
    parser.add_argument('--test_method', default='five', type=str,
                        choices=['one', 'five'],
                        help='Test method: one crop / five crop (default: five)')
    parser.add_argument('--output_dir', default='results', type=str,
                        help='Directory to save results')
    parser.add_argument('--gt_meta_dir', default='metas', type=str,
                        help='Directory of ground truth JSON files')
    
    args = parser.parse_args()
    return args

def get_image_paths(dataset_root, dataset_name):
    """Get all image paths of the specified dataset"""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not exist: {dataset_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def load_ground_truth(dataset_name, meta_dir):
    """Load ground truth scores (cache as dictionary for fast lookup)"""
    gt_path = os.path.join(meta_dir, f"{dataset_name}.json")
    if not os.path.exists(gt_path):
        raise ValueError(f"Ground truth file not found: {gt_path}")
    
    with open(gt_path, 'r') as f:
        data = json.load(f)
    gt_dict = {item['image']: item['score'] for item in data}
    return gt_dict

def process_dataset(model, device, dataset_name, image_paths, test_method, output_file):
    """Process a single dataset and return predictions and valid paths"""
    model.eval()
    all_scores = []
    valid_paths = []

    # Define image preprocessing
    if test_method == 'one':
        transformations = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transformations = transforms.Compose([
            transforms.Resize(384),
            transforms.FiveCrop(320),
            lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
            lambda crops: torch.stack([transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])(crop) for crop in crops])
        ])

    # Process images with no gradient computation
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f"Processing {dataset_name}"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transformations(img)
                img_tensor = img_tensor.to(device)

                # Model prediction: solve return value mismatch problem
                model_output = model(img_tensor)
                if isinstance(model_output, tuple):
                    outputs = model_output[-1]
                else:
                    outputs = model_output

                # Handle test method (one crop / five crop)
                if test_method == 'one':
                    score = outputs.item() if outputs.numel() == 1 else outputs.mean().item()
                else:
                    score = outputs.mean().item()

                all_scores.append(score)
                valid_paths.append(img_path)

                # Save individual score
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{img_path},{score:.6f}\n")

            except Exception:
                # Skip redundant error print to keep terminal clean
                continue

    # Save average score
    if all_scores:
        avg_score = np.mean(all_scores)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{dataset_name} average score: {avg_score:.6f}\n")

    return all_scores, valid_paths

def calculate_metrics(y_true, y_pred):
    """Calculate SRCC and PLCC using numpy for fast computation"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    srcc, _ = stats.spearmanr(y_true, y_pred)
    plcc, _ = stats.pearsonr(y_true, y_pred)
    return srcc, plcc

if __name__ == '__main__':
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (no DataParallel)
    model = stairIQA_resnet.resnet50(pretrained=False)
    model = model.to(device)

    # Load model weights: optimize prefix removal + strict loading
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        # Remove module. prefix
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    except Exception:
        # Fallback to non-strict mode without print
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    # Process each test dataset
    for dataset in args.test_datasets:
        image_paths = get_image_paths(args.dataset_root, dataset)

        if not image_paths:
            continue

        output_file = os.path.join(args.output_dir, f"{dataset}_results.csv")
        # Initialize result file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Image Path,Prediction Score\n")

        # Get predictions
        pred_scores, valid_paths = process_dataset(
            model, device, dataset, image_paths, args.test_method, output_file
        )

        if not pred_scores:
            continue

        # Load ground truth and calculate metrics
        try:
            gt_dict = load_ground_truth(dataset, args.gt_meta_dir)
        except ValueError:
            continue

        # Match predictions with ground truth
        y_true = []
        y_pred = []
        for img_path, pred in zip(valid_paths, pred_scores):
            rel_path = os.path.relpath(img_path, args.dataset_root)
            rel_path = rel_path.replace(os.sep, '/')
            if rel_path in gt_dict:
                y_true.append(gt_dict[rel_path])
                y_pred.append(pred)

        # Calculate and save SRCC/PLCC
        if len(y_true) >= 2:
            srcc, plcc = calculate_metrics(y_true, y_pred)
            # Only keep core metric print (the most important output)
            print(f"\n{dataset} - SRCC: {srcc:.6f}, PLCC: {plcc:.6f}")
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"\nSRCC: {srcc:.6f}\n")
                f.write(f"PLCC: {plcc:.6f}\n")

    print("\n========== Evaluation Completed ==========")
    print(f"Results saved in: {args.output_dir}")