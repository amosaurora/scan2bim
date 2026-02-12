# import numpy as np
# from os import listdir, path

# np.random.seed(12345)

# dpath = "O:/data/S3DIS/"
# # dpath = "O:/data/S3DIS/"
# fnames = [f for f in listdir(dpath) if path.isfile(path.join(dpath,f)) and not f.startswith('.')]
# per = np.random.permutation(len(fnames))
# fnames = [fnames[i] for i in per]

# with open("train.txt", "w") as f:
#     for i in range(0, len(fnames)-len(fnames)//5):
#         f.write(fnames[i]+"\n")

# with open("val.txt", "w") as f:
#     for i in range(len(fnames)-len(fnames)//5, len(fnames)-len(fnames)//10):
#         f.write(fnames[i]+"\n")

# with open("test.txt", "w") as f:
#     for i in range(len(fnames)-len(fnames)//10, len(fnames)):
#         f.write(fnames[i]+"\n")

import os
import random
import glob
import argparse

# --- CONFIGURATION ---
DATA_PATH = "O:/data/S3DIS"  # Change to your actual folder path

# Keywords to identify file types
SYNTH_KEYWORD = "synth"     # e.g., "synth_room_001.ply"
CUSTOM_KEYWORD = "Classified"   # e.g., "custom_office_scan.ply"
S3DIS_PREFIX = "Area_"      # e.g., "Area_1_conferenceRoom_1.ply"

# How much to repeat them?
REPEAT_SYNTH = 5      # 200 * 5 = 1,000 samples
REPEAT_CUSTOM = 100   # 9 * 100 = 900 samples
# ---------------------

def main():
    # 1. Get all PLY files in the folder
    all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.ply')]
    print(f"Found {len(all_files)} total files.")

    s3dis_files = []
    synth_files = []
    custom_files = []
    others = []

    # 2. Sort files into lists
    for f in all_files:
        if SYNTH_KEYWORD in f.lower():
            synth_files.append(f)
        elif CUSTOM_KEYWORD in f.lower(): # Or match your specific custom filenames
            custom_files.append(f)
        elif f.startswith(S3DIS_PREFIX):
            s3dis_files.append(f)
        else:
            others.append(f)

    print(f"  - S3DIS Files: {len(s3dis_files)}")
    print(f"  - Synthetic Files: {len(synth_files)}")
    print(f"  - Custom Files: {len(custom_files)}")
    if others:
        print(f"  - Unclassified Files: {len(others)} (Check naming!)")

    # 3. Create Train/Test Split (Only split S3DIS)
    # We put ALL synthetic and custom files into TRAIN.
    # We reserve 10% of S3DIS for testing.
    random.shuffle(s3dis_files)
    split_idx = int(len(s3dis_files) * 0.8)
    
    train_s3dis = s3dis_files[:split_idx]
    val_s3dis = s3dis_files[split_idx:int(len(s3dis_files)*0.9)]
    test_s3dis = s3dis_files[split_idx:]

    # 4. Generate Train List (With Oversampling)
    train_list = []
    
    # Add S3DIS (1x)
    train_list.extend(train_s3dis)
    
    # Add Synthetic (5x)
    for _ in range(REPEAT_SYNTH):
        train_list.extend(synth_files)
        
    # Add Custom (100x)
    for _ in range(REPEAT_CUSTOM):
        train_list.extend(custom_files)
        
    # Shuffle the final training list so batches are mixed
    random.shuffle(train_list)

    # 5. Write to files
    with open(os.path.join(DATA_PATH, "train.txt"), "w") as f:
        for filename in train_list:
            f.write(filename + "\n")
            
    with open(os.path.join(DATA_PATH, "test.txt"), "w") as f:
        for filename in test_s3dis:
            f.write(filename + "\n")

    with open(os.path.join(DATA_PATH, "val.txt"), "w") as f:
            for filename in val_s3dis:
                f.write(filename + "\n")

    print("="*30)
    print(f"DONE!")
    print(f"Train samples: {len(train_list)} (S3DIS + {REPEAT_SYNTH}x Synth + {REPEAT_CUSTOM}x Custom)")
    print(f"Validation samples: {len(val_s3dis)}")
    print(f"Test samples:  {len(test_s3dis)} (Pure S3DIS)")

if __name__ == "__main__":
    main()