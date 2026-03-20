import numpy as np
import glob
import os
from plyfile import PlyData

# Define your dataset path here
DATASET_PATH = "Q:\S3DIS\camera_view" 

def calculate_class_weights(dataset_path):
    # Initialize counter for 8 classes (0 to 7)
    # Mapping: 0:ceil, 1:floor, 2:wall, 3:beam, 4:col, 5:win, 6:door, 7:clutter
    class_counts = np.zeros(8, dtype=np.int64)
    
    # Find all PLY files
    files = glob.glob(os.path.join(dataset_path, "*.ply"))
    print(f"Found {len(files)} ply files. Scanning...")

    for f in files:
        try:
            ply = PlyData.read(f)
            # stored as 'label' or 'scalar_Label' depending on software
            # We assume 'label' based on your previous messages
            if 'label' in ply['vertex']:
                labels = ply['vertex']['label']
            elif 'scalar_label' in ply['vertex']:
                labels = ply['vertex']['scalar_label']
            else:
                print(f"Skipping {f}: No 'label' field found.")
                continue

            labels = np.round(labels).astype(np.int32)
                
            # Count occurrences of each label
            unique, counts = np.unique(labels, return_counts=True)
            
            for u, c in zip(unique, counts):
                if 0 <= u < 8:
                    class_counts[int(u)] += c
                    
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    
    class_names = ["Ceiling", "Floor", "Wall", "Beam", "Column", "Window", "Door", "Unassigned"]
    
    print("Class Counts (n):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name:<12} : {class_counts[i]}")

    # Calculate Weights: w = 1 / n
    # Add epsilon to prevent division by zero if a class is missing
    epsilon = 1e-6
    weights = 1.0 / (class_counts + epsilon)
    
    # Normalize weights so they sum to similar magnitude (optional but good for stability)
    # But usually raw 1/n is fine as long as you use the sqrt in the loss function later.
    
    print("\nCOPY THIS ARRAY INTO S3DISdataset.py:")
    print("pts = np.array(" + str(class_counts.tolist()) + ", dtype=np.int32)")
    
    return class_counts

if __name__ == "__main__":
    calculate_class_weights(DATASET_PATH)