import numpy as np
import glob
import os
from plyfile import PlyData
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_PATH = "O:/data/S3DIS"       # Current folder
REPEAT_SYNTH = 5      # Must match makelists.py
REPEAT_CUSTOM = 100   # Must match makelists.py

# Keywords to match files
SYNTH_KEYWORD = "synth"
CUSTOM_KEYWORD = "Classified"
S3DIS_PREFIX = "Area_"

# Mapping: Original S3DIS (0-13) -> Training Classes (0-7)
REMAP_DICT = {
    0: 0,   # ceiling
    1: 1,   # floor
    2: 2,   # wall
    3: 3,   # beam
    4: 4,   # column
    5: 5,   # window
    6: 6,   # door
    7: 7,   # unassigned
    8: 7,   # table -> unassigned
    9: 7,   # chair -> unassigned
    10: 7,  # sofa -> unassigned
    11: 7,  # bookcase -> unassigned
    12: 7,  # board -> unassigned
    13: 7   # clutter -> unassigned
}

def get_class_counts(file_path):
    """Reads a PLY file and returns class counts (0-7)."""
    try:
        with open(file_path, 'rb') as f:
            ply = PlyData.read(f)
            
        # 1. Find the label field
        if 'label' in ply['vertex']:
            labels = ply['vertex']['label']
        elif 'scalar_label' in ply['vertex']:
            labels = ply['vertex']['scalar_label']
        elif 'scalar_Classification' in ply['vertex']:
            labels = ply['vertex']['scalar_Classification']
        else:
            return None

        # 2. Remap labels (0-13 -> 0-7)
        labels = np.round(labels).astype(int)
        
        # Fast Vectorized Remap
        # Handle labels outside 0-13 safely
        labels = np.clip(labels, 0, 13) 
        remapped = np.vectorize(REMAP_DICT.get)(labels)
        
        # 3. Count
        counts = np.bincount(remapped, minlength=8)
        return counts[:8] # Ensure exactly 8 classes

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.zeros(8, dtype=int)

def main():
    ply_files = glob.glob(os.path.join(DATA_PATH, "*.ply"))
    print(f"Found {len(ply_files)} total files. Scanning labels...")
    
    total_counts = np.zeros(8, dtype=np.int64)
    
    # Iterate and sum up
    for f in tqdm(ply_files):
        # 1. Get counts for this file
        counts = get_class_counts(f)
        if counts is None: continue
            
        # 2. Determine Multiplier
        fname = os.path.basename(f).lower()
        if SYNTH_KEYWORD in fname:
            multiplier = REPEAT_SYNTH
        elif CUSTOM_KEYWORD in fname:
            multiplier = REPEAT_CUSTOM
        elif fname.startswith(S3DIS_PREFIX.lower()):
            multiplier = 1
        else:
            multiplier = 1 # Default for unknown files
            
        # 3. Add to total
        total_counts += (counts * multiplier)

    print("\n" + "="*40)
    print("NEW CLASS WEIGHTS (Copy this!)")
    print("="*40)
    
    # Format nicely for Python
    array_str = ", ".join(map(str, total_counts))
    print(f"pts = np.array([{array_str}], dtype=np.int32)")
    
    # Calculate expected weights to verify
    weights = 1.0 / (total_counts + 1e-6)
    med = np.median(weights)
    print("\nSanity Check (Normalized Weights):")
    for i, w in enumerate(weights):
        print(f" Class {i}: {w/med:.4f} x Median")

if __name__ == "__main__":
    main()