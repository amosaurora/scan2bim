import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
from sklearn.neighbors import NearestNeighbors # Added for smoothing

import torch
torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.segcloud import SegCloud
from model.bimnet import BIMNet
from dataloaders.PCSdataset import PCSDataset
from dataloaders.S3DISdataset import S3DISDataset

from sklearn.cluster import DBSCAN
import json
from pathlib import Path
from matplotlib.colors import to_rgb
import argparse

def load_point_cloud(file_path):
    """
    Load point cloud from file.
    Supports: .ply, .pcd, .xyz, .las formats
    """
    print(f"Loading point cloud from: {file_path}")
    
    # Open3D can read .ply and .pcd directly
    if file_path.suffix in ['.ply', '.pcd']:
        pcd = o3d.io.read_point_cloud(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"Loaded {len(pcd.points)} points")
    return pcd

# Map label IDs -> semantic names (match your training setup)
ID_TO_NAME = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "unassigned",
}

def separate_by_label(pcd, point_labels):
    """
    Split point cloud into semantic classes using integer labels, not colors.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    separated = {}
    for class_id, class_name in ID_TO_NAME.items():
        mask = (point_labels == class_id)
        if not np.any(mask):
            continue

        class_pcd = o3d.geometry.PointCloud()
        class_pcd.points = o3d.utility.Vector3dVector(points[mask])
        class_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        separated[class_name] = class_pcd
        print(f"  {class_name}: {mask.sum()} points")

    return separated

def smooth_labels_knn(pcd, labels, k=5):
    """
    Replaces each point's label with the majority label of its k-nearest neighbors.
    Removes 'salt-and-pepper' noise.
    """
    print(f"Smoothing labels with KNN (k={k})...")
    points = np.asarray(pcd.points)
    
    # 1. Build Neighbor Tree
    # n_jobs=-1 uses all CPU cores for speed
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # 2. Vote
    new_labels = np.zeros_like(labels)
    
    # Vectorized voting (faster than loop)
    neighbor_labels = labels[indices] # Shape [N, k]
    
    # Use scipy mode if available, else simple loop
    from scipy.stats import mode
    try:
        # mode returns (values, counts)
        vote_result = mode(neighbor_labels, axis=1, keepdims=False)
        new_labels = vote_result[0]
    except:
        # Fallback if scipy version differs
        for i in tqdm(range(len(labels)), desc="Voting"):
            counts = np.bincount(neighbor_labels[i])
            new_labels[i] = np.argmax(counts)
            
    return new_labels

def instantiate_with_dbscan(pcd, class_name, eps=0.1, min_points=100):
    """
    Use DBSCAN clustering to identify individual instances within a class.
    """
    if len(pcd.points) == 0:
        return []
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"\nClustering {class_name} with DBSCAN...")
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit(points)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    print(f"  Found {n_clusters} instances")
    
    instances = []
    for label_id in unique_labels:
        if label_id == -1:  # Skip noise points
            continue
        
        instance_mask = labels == label_id
        instance_points = points[instance_mask]
        instance_colors = colors[instance_mask]
        
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(instance_points)
        instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)
        
        instances.append(instance_pcd)
    
    return instances

def filter_small_instances(instances_dict, min_points_thresholds):
    """
    Removes instances that are too small to be real BIM objects.
    """
    cleaned_dict = {}
    print("\n--- CLEANING NOISE ---")
    
    for class_name, instances in instances_dict.items():
        thresh = min_points_thresholds.get(class_name, 500)
        
        valid_instances = []
        for i, pcd in enumerate(instances):
            n_points = len(pcd.points)
            if n_points >= thresh:
                valid_instances.append(pcd)
        
        cleaned_dict[class_name] = valid_instances
        removed = len(instances) - len(valid_instances)
        if removed > 0:
            print(f"  {class_name}: Removed {removed} small instances (<{thresh} pts)")
            
    return cleaned_dict

def save_instances(instances_dict, output_dir):
    """
    Save all instantiated point clouds to separate files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_name, instances in instances_dict.items():
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i, instance in enumerate(instances):
            filename = class_dir / f"{class_name}_instance_{i:03d}.ply"
            o3d.io.write_point_cloud(str(filename), instance)
        
    # Also save summary as JSON
    summary = {
        class_name: len(instances) 
        for class_name, instances in instances_dict.items()
    }
    
    with open(output_path / "instantiation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_path / 'instantiation_summary.json'}")

    combined_pc = o3d.geometry.PointCloud()
    for class_name, instances in instances_dict.items():
        for instance in instances:
            combined_pc += instance   # merge point clouds
    
    combined_filename = output_path / "all_instances_combined.ply"
    o3d.io.write_point_cloud(str(combined_filename), combined_pc)
    print(f"Combined point cloud saved to {combined_filename}")

def generate_distinct_colors(n_colors):
    try: 
        cmap = plt.colormaps['tab20']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab20')
    
    colors = []
    for i in range(n_colors):
        rgba = cmap(i / max(n_colors, 1))
        colors.append(rgba[:3]) 
    return colors

def visualize_instances(instances_dict, show_by_class=True):
    if show_by_class:
        for class_name, instances in instances_dict.items():
            if len(instances) == 0:
                continue
            
            print(f"\nVisualizing {class_name} instances ({len(instances)} instances)...")
            instance_colors = generate_distinct_colors(len(instances))
            colored_instances = []
            for i, instance in enumerate(instances):
                colored_pcd = o3d.geometry.PointCloud(instance)
                instance_color = np.tile(instance_colors[i], (len(instance.points), 1))
                colored_pcd.colors = o3d.utility.Vector3dVector(instance_color)
                colored_instances.append(colored_pcd)
            
            o3d.visualization.draw_geometries(colored_instances,
                                            window_name=f"{class_name} - {len(instances)} Instances",
                                            width=1024, height=768)
    else:
        print("\nVisualizing all instances from all classes...")
        all_colored_instances = []
        for class_name, instances in instances_dict.items():
            if len(instances) == 0: continue
            instance_colors = generate_distinct_colors(len(instances))
            for i, instance in enumerate(instances):
                colored_pcd = o3d.geometry.PointCloud(instance)
                instance_color = np.tile(instance_colors[i], (len(instance.points), 1))
                colored_pcd.colors = o3d.utility.Vector3dVector(instance_color)
                all_colored_instances.append(colored_pcd)
        
        if all_colored_instances:
            o3d.visualization.draw_geometries(all_colored_instances,
                                            window_name=f"All Instances",
                                            width=1024, height=768)

def visualize_summary(instances_dict, separated_classes, original_pcd):
    print("\n" + "=" * 60)
    print("VISUALIZATION MODE")
    print("=" * 60)
    # o3d.visualization.draw_geometries([original_pcd], window_name="Original", width=800, height=600)
    
    if separated_classes:
        o3d.visualization.draw_geometries(list(separated_classes.values()), window_name="Semantic Classes", width=800, height=600)
    
    if instances_dict:
        visualize_instances(instances_dict, show_by_class=False)
    
    # Ask if user wants to see all instances separately
    print("\n" + "=" * 60)
    response = input("Would you like to see all instances from all classes separately? (y/n): ")
    if response.lower() == 'y':
        visualize_instances(instances_dict, show_by_class=True)

def finetune_model(checkpoint_path, device, num_old_classes, num_new_classes):
    state_old = torch.load(checkpoint_path, map_location=device)
    model_new = BIMNet(num_classes=num_new_classes)
    state_new = model_new.state_dict()

    transferred, skipped = [], []
    for k, v in state_old.items():
        if k in state_new and state_new[k].shape == v.shape:
            state_new[k] = v
            transferred.append(k)
        else:
            skipped.append(k)
    
    if skipped:
        print("Skipped parameters:")
        for k in skipped:
            print(f" - {k} : {state_old[k].shape}")

    model_new.load_state_dict(state_new)
    model_new.to(device)
    model_new.train()

    return model_new

def build_models(checkpoint_paths, device, num_classes=8):
    models = []
    for ckpt in checkpoint_paths:
        print(f"Loading checkpoint: {ckpt}")
        model = finetune_model(ckpt, device, num_old_classes=13, num_new_classes=num_classes)
        # state = torch.load(ckpt, map_location=device)
        # model.load_state_dict(state)
        # model.to(device)
        model.eval()
        models.append(model)
    return models

def voxelize_points(points, cube_edge):
    points_centered = points - points.mean(axis=0)
    
    max_val = np.abs(points_centered).max() + 1e-8
    points_norm = points_centered / max_val
    
    points_shifted = points_norm + 1.0
    
    scale_factor = cube_edge // 2
    points_grid = np.round(points_shifted * scale_factor).astype(np.int32)
    
    points_grid = np.clip(points_grid, 0, cube_edge - 1)

    vox = np.zeros((1, cube_edge, cube_edge, cube_edge), dtype=np.float32)
    vox[0, points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]] = 1.0

    return vox, points_grid

def color_label(labels, num_classes=8):
    cmap = plt.get_cmap("tab20", num_classes)
    flat = labels.flatten()
    colors = cmap(flat % num_classes)[:, :3] 
    return colors.reshape((*labels.shape, 3))

def run_bimnet_inference(pcd, models, cube_edge=128, num_classes=8, device="cuda"):
    points = np.asarray(pcd.points)
    print(f"Loaded {points.shape[0]} points")
    vox, points_grid = voxelize_points(points, cube_edge)
    x = torch.from_numpy(vox).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_sum = None
        for model in models:
            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_avg = logits_sum / len(models)
        preds = logits_avg.argmax(dim=1).squeeze(0).cpu().numpy()

    colors_volume = color_label(preds, num_classes=num_classes)
    point_colors = colors_volume[points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]]
    point_labels = preds[points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]]

    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    return pcd, preds, points_grid, point_labels

def instantiate_planar_iterative(pcd, class_name, dist_thresh=0.20, min_points=500):
    """
    Separates planar instances (Walls/Floors) by iteratively finding planes 
    and removing them from the cloud until no valid planes remain.
    UPDATED: dist_thresh increased to 0.20 to fix fragmented walls.
    """
    remaining_pcd = pcd
    instances = []
    
    print(f"\nIterative RANSAC for {class_name} (Thresh={dist_thresh})...")
    
    while len(remaining_pcd.points) > min_points:
        points = np.asarray(remaining_pcd.points)
        
        # Fit a single plane
        plane = pyrsc.Plane()
        # Note: pyransac3d returns equation (4 floats) and inliers (indices)
        best_eq, inliers = plane.fit(points, thresh=dist_thresh, minPoints=100, maxIteration=1000)
        
        # If not enough inliers, stop
        if len(inliers) < min_points:
            break
            
        # Extract the instance
        inst_pcd = remaining_pcd.select_by_index(inliers)
        inst_pcd.paint_uniform_color(generate_distinct_colors(len(instances)+1)[-1])
        instances.append(inst_pcd)
        
        # Remove these points and continue
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        print(f"  Found instance {len(instances)}: {len(inliers)} points. Remaining: {len(remaining_pcd.points)}")
        
    return instances

def extract_bim_parameters(instances_dict):
    """
    Calculates BIM-ready parameters (Length, Height, Thickness, Centerline) 
    for each wall instance.
    """
    bim_data = []
    
    for class_name, pcd_list in instances_dict.items():
        if class_name != "wall": 
            continue 
            
        for idx, pcd in enumerate(pcd_list):
            pts = np.asarray(pcd.points)
            if len(pts) < 50: continue

            # 1. Height (Z-axis extent)
            z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
            height = z_max - z_min
            
            # 2. Centerline (2D Projection on XY plane)
            xy_pts = pts[:, :2]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca.fit(xy_pts)
            
            direction = pca.components_[0] 
            center = xy_pts.mean(axis=0)
            
            projected = xy_pts @ direction
            p_min, p_max = projected.min(), projected.max()
            
            start_pt = center + direction * (p_min - projected.mean())
            end_pt = center + direction * (p_max - projected.mean())
            
            # 3. Thickness (Requires finding the parallel surface or assuming standard)
            thickness = 0.2 
            
            bim_obj = {
                "id": f"{class_name}_{idx}",
                "type": class_name,
                "height": float(height),
                "thickness": float(thickness),
                "geometry": {
                    "start_x": float(start_pt[0]), "start_y": float(start_pt[1]), "start_z": float(z_min),
                    "end_x": float(end_pt[0]), "end_y": float(end_pt[1]), "end_z": float(z_min)
                }
            }
            bim_data.append(bim_obj)
            
    return bim_data

def main(
    input_file,
    output_dir="output_instances",
    checkpoint_paths=None,
    cube_edge=96,
    num_classes=8,
    device=None,
    visualize_network_output=False,
    visualize_instances_flag=False
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = checkpoint_paths 
    
    print("=" * 60)
    print("Point Cloud Instantiation Workflow (BIMNet + DBSCAN)")
    print("=" * 60)

    # Step 1: Load point cloud
    input_path = Path(input_file)
    pcd = load_point_cloud(input_path)

    # Step 2: Load models and run BIMNet
    print("\nLoading BIMNet models...")
    models = build_models(checkpoint_paths, device, num_classes=num_classes)

    print("\nRunning BIMNet inference...")
    pcd, preds_volume, points_grid, point_labels = run_bimnet_inference(
        pcd, models, cube_edge=cube_edge, num_classes=num_classes, device=device
    )

    # --- NEW STEP: SMOOTH LABELS ---
    # Fixes salt-and-pepper noise before any separation happens
    print("\nStep 0.5: Smoothing predictions with KNN...")
    point_labels = smooth_labels_knn(pcd, point_labels, k=10)
    
    # if visualize_network_output:
    #     print("\nVisualizing BIMNet semantic prediction...")
    #     o3d.visualization.draw_geometries([pcd])

    # Step 3: Separate by semantic class (color)
    print("\nStep 1: Separating point cloud by semantic class...")
    separated_classes = separate_by_label(pcd, point_labels)

    if not separated_classes:
        print("Warning: No classes found! Check your color mappings.")
        return None

    # Step 4: Instantiate each class using DBSCAN / RANSAC
    print("\nStep 2: Instantiating classes...")
    all_instances = {}

    planar_classes = ['wall', 'floor', 'ceiling']
    
    dbscan_params = {
        'beam':      {'eps': 0.1, 'min_points': 150},
        'column':    {'eps': 0.1, 'min_points': 125},
        'window':    {'eps': 0.05, 'min_points': 150},
        'door':      {'eps': 0.07, 'min_points': 200},
        'unassigned': {'eps': 0.03, 'min_points': 300},
    }

    for class_name, class_pcd in separated_classes.items():
        if class_name in planar_classes:
            # UPDATED: Thresh 0.20 handles wavy walls
            instances = instantiate_planar_iterative(class_pcd, class_name, dist_thresh=0.20)
        else:
            params = dbscan_params.get(class_name, {'eps': 0.1, 'min_points': 100})
            instances = instantiate_with_dbscan(
                class_pcd,
                class_name,
                eps=params['eps'],
                min_points=params['min_points'],
            )
        all_instances[class_name] = instances

    # UPDATED: Aggressive thresholds to delete ghost instances
    cleaning_thresholds = {
        'ceiling': 3000, 
        'floor': 3000,   
        'wall': 1500,
        'beam': 300,
        'column': 300,
        'door': 1000, 
        'window': 300,
        'unassigned': 100
    }

    all_instances = filter_small_instances(all_instances, cleaning_thresholds)

    # Step 5: Extract BIM Data & Save
    print("\nStep 3: Extracting BIM Parameters and Saving...")
    save_instances(all_instances, output_dir)
    
    bim_json_data = extract_bim_parameters(all_instances)
    with open(Path(output_dir) / "bim_reconstruction_data.json", "w") as f:
        json.dump(bim_json_data, f, indent=4)
    print(f"BIM parameters saved to {output_dir}/bim_reconstruction_data.json")

    # Step 6: Optional visualization
    if visualize_instances_flag:
        visualize_summary(all_instances, separated_classes, pcd)

    return all_instances, separated_classes, pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BIMNet semantic segmentation + DBSCAN instance extraction"
    )
    parser.add_argument("--input_file", help="Path to input point cloud (.ply/.pcd)")
    parser.add_argument("--output_dir", default="output_instances", help="Directory to save instance PLYs")
    parser.add_argument("--checkpoint", action="append", default=[], help="Path(s) to BIMNet checkpoint(s)")
    parser.add_argument("--cube_edge", type=int, default=96, help="Voxel grid edge length")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of BIMNet output classes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--vis-net", action="store_true", help="Visualize BIMNet output")
    parser.add_argument("--vis-instances", action="store_true", help="Visualize DBSCAN instances")
    
    args = parser.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    main(
        input_file=args.input_file,
        output_dir=args.output_dir,
        checkpoint_paths=args.checkpoint,
        cube_edge=args.cube_edge,
        num_classes=args.num_classes,
        device=device,
        visualize_network_output=args.vis_net,
        visualize_instances_flag=args.vis_instances
    )