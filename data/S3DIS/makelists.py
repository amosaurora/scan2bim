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
import argparse

# --- CONFIGURATION ---
DATA_PATH = "O:/data/S3DIS"  # Change to your actual folder path
RANDOM_SEED = 12345

# Keywords to identify file types
SYNTH_KEYWORD = "synth_"     # e.g., "synth_room_001.ply"
CUSTOM_KEYWORD = "classified"   # e.g., "custom_office_scan.ply"
S3DIS_PREFIX = "Area_"      # e.g., "Area_1_conferenceRoom_1.ply"

DEFAULT_REPEAT_SYNTH = 2
DEFAULT_REPEAT_CUSTOM = 8
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="folder containing .ply files")
    parser.add_argument("--mode", choices=["mixed", "synthetic_custom", "custom_only"], default="mixed",
                        help="which domains to include in train.txt")
    parser.add_argument("--repeat_synth", type=int, default=DEFAULT_REPEAT_SYNTH, help="synthetic oversampling factor")
    parser.add_argument("--repeat_custom", type=int, default=DEFAULT_REPEAT_CUSTOM, help="custom oversampling factor")
    parser.add_argument("--custom_train_ratio", type=float, default=0.8, help="fraction of custom files used for train")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    data_path = args.data_path

    # 1. Get all PLY files in the folder
    all_files = [f for f in os.listdir(data_path) if f.endswith('.ply')]
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
        print(f"    Example: {others[:3]}")

    # 3. Create Train/Test Split (Only split S3DIS)
    random.shuffle(s3dis_files)
    random.shuffle(custom_files)
    
    n_total = len(s3dis_files)
    idx_80 = int(n_total * 0.8)
    idx_90 = int(n_total * 0.9)
    
    # 0% to 80% -> Train
    train_s3dis = s3dis_files[:idx_80]
    # 80% to 90% -> Val
    val_s3dis = s3dis_files[idx_80:idx_90]
    # 90% to 100% -> Test
    test_s3dis = s3dis_files[idx_90:]

    custom_split_idx = int(len(custom_files) * args.custom_train_ratio)
    if len(custom_files) > 1:
        custom_split_idx = min(max(custom_split_idx, 1), len(custom_files) - 1)
    train_custom = custom_files[:custom_split_idx]
    val_custom = custom_files[custom_split_idx:]
    if len(custom_files) == 1:
        train_custom = custom_files
        val_custom = custom_files

    # 4. Generate Train List (With Oversampling)
    train_list = []

    if args.mode == "mixed":
        train_list.extend(train_s3dis)
        for _ in range(args.repeat_synth):
            train_list.extend(synth_files)
        for _ in range(args.repeat_custom):
            train_list.extend(train_custom)
    elif args.mode == "synthetic_custom":
        for _ in range(args.repeat_synth):
            train_list.extend(synth_files)
        for _ in range(args.repeat_custom):
            train_list.extend(train_custom)
    elif args.mode == "custom_only":
        for _ in range(args.repeat_custom):
            train_list.extend(train_custom)
        
    # Shuffle the final training list so batches are mixed
    random.shuffle(train_list)

    # 5. Write to files
    with open(os.path.join(data_path, "train.txt"), "w") as f:
        for filename in train_list:
            f.write(filename + "\n")
            
    with open(os.path.join(data_path, "test.txt"), "w") as f:
        for filename in test_s3dis:
            f.write(filename + "\n")

    with open(os.path.join(data_path, "val.txt"), "w") as f:
            for filename in val_s3dis:
                f.write(filename + "\n")

    with open(os.path.join(data_path, "val_custom.txt"), "w") as f:
        for filename in val_custom:
            f.write(filename + "\n")

    print("="*30)
    print(f"DONE!")
    print(f"Mode: {args.mode}")
    print(f"Train samples: {len(train_list)}")
    print(f"  repeat_synth={args.repeat_synth}")
    print(f"  repeat_custom={args.repeat_custom}")
    print(f"Validation samples: {len(val_s3dis)}")
    print(f"Custom validation samples: {len(val_custom)}")
    print(f"Test samples:  {len(test_s3dis)} (Pure S3DIS)")

if __name__ == "__main__":
    main()
