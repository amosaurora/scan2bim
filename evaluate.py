import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import os

# Import your project modules
from model.bimnet import BIMNet
from dataloaders.S3DISdataset import S3DISDataset
from util.metrics import Metrics 

def evaluate(model, loader, device, split_name):
    """
    Runs inference on the loader and calculates metrics.
    """
    # Initialize metrics (skipping class 0 if your train script does, 
    # but usually we want to see all classes. Adjust cnames[1:] if needed)
    # Based on your train script:
    metric = Metrics(loader.dataset.cnames[1:], device=device)
    
    model.eval()
    print(f"\n--- Evaluating on {split_name} set ---")
    
    with torch.no_grad():
        # Iterate over the dataset
        for x, y in tqdm(loader, total=len(loader)):
            x = x.to(device)
            # CRITICAL: Match the training script's index shift
            y = y.to(device, dtype=torch.long) - 1 
            
            # Forward pass
            outputs = model(x)
            
            # Predictions
            preds = outputs.argmax(dim=1).flatten()
            targets = y.flatten()
            
            # Update metrics
            metric.add_sample(preds, targets)
            
    # Calculate final results
    miou = metric.percent_mIoU()
    acc = metric.percent_acc()
    prec = metric.percent_prec()
    
    print("\n" + "="*30)
    print(f"RESULTS: {split_name.upper()}")
    print("="*30)
    print(f"Mean IoU:        {miou:.2f}%")
    print(f"Point Accuracy:  {acc:.2f}%")
    print(f"Point Precision: {prec:.2f}%")
    print("-" * 30)
    
    # Print per-class IoU
    ious = metric.IoU()
    names = metric.name_classes
    print("Per-Class IoU:")
    for name, iou in zip(names, ious):
        print(f"  {name:<12}: {iou*100:.2f}%")
    print("="*30 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=['train', 'val', 'test'], help="Which set to test on")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth model file")
    parser.add_argument("--dset_path", type=str, default="Scan-to-BIM", help="Path to dataset root")
    parser.add_argument("--cube_edge", type=int, default=96, help="Voxel size (must match training)")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Dataset
    print(f"Loading {args.split} dataset...")
    # Note: 'augment=False' is important for testing to get consistent results
    dset = S3DISDataset(root_path=args.dset_path, 
                        split=args.split, 
                        cube_edge=args.cube_edge, 
                        augment=False)
    
    loader = DataLoader(dset, 
                        batch_size=1,  # Use batch_size 1 for testing
                        shuffle=False, 
                        num_workers=4)

    # 2. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = BIMNet(num_classes=args.num_classes)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # 3. Run Evaluation
    evaluate(model, loader, device, args.split)