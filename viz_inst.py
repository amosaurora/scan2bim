import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyransac3d as pyrsc

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
        # For other formats, you might need to convert or use different libraries
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

    pcd          : Open3D point cloud with all points
    point_labels : np.ndarray [N] of integer label IDs (0..7)
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

def instantiate_with_dbscan(pcd, class_name, eps=0.1, min_points=100):
    """
    Use DBSCAN clustering to identify individual instances within a class.
    
    Parameters:
    - pcd: Point cloud for one semantic class
    - class_name: Name of the class (for logging)
    - eps: Maximum distance between points in same cluster (adjust based on your scale)
    - min_points: Minimum points to form a cluster
    
    Returns: List of point clouds, each representing one instance
    """
    if len(pcd.points) == 0:
        return []
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"\nClustering {class_name} with DBSCAN...")
    print(f"  Total points: {len(points)}")
    print(f"  Parameters: eps={eps}, min_points={min_points}")
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = clustering.labels_
    
    # Count instances (excluding noise points labeled as -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"  Found {n_clusters} instances")
    if n_noise > 0:
        print(f"  {n_noise} points marked as noise (not part of any instance)")
    
    # Create separate point cloud for each instance
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

def save_instances(instances_dict, output_dir):
    """
    Save all instantiated point clouds to separate files.
    
    instances_dict: Dictionary with structure {class_name: [instance1, instance2, ...]}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_name, instances in instances_dict.items():
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i, instance in enumerate(instances):
            filename = class_dir / f"{class_name}_instance_{i:03d}.ply"
            o3d.io.write_point_cloud(str(filename), instance)
        
        print(f"Saved {len(instances)} instances of {class_name} to {class_dir}")
    
    # Also save summary as JSON
    summary = {
        class_name: len(instances) 
        for class_name, instances in instances_dict.items()
    }
    
    with open(output_path / "instantiation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_path / 'instantiation_summary.json'}")

def generate_distinct_colors(n_colors):
    """
    Generate n distinct colors for visualization.
    Uses a colormap to ensure colors are visually distinct.
    """
    try:
        # Try new matplotlib API (v3.5+)
        cmap = plt.colormaps['tab20']
    except (AttributeError, KeyError):
        # Fallback to old API
        cmap = plt.cm.get_cmap('tab20')
    
    colors = []
    for i in range(n_colors):
        rgba = cmap(i / max(n_colors, 1))
        colors.append(rgba[:3])  # RGB only, ignore alpha
    return colors

def visualize_original_pointcloud(pcd):
    """
    Visualize the original point cloud.
    """
    print("\nVisualizing original point cloud...")
    print("Close the window to continue.")
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="Original Point Cloud",
                                      width=1024, 
                                      height=768)

def visualize_separated_classes(separated_classes):
    """
    Visualize all separated semantic classes together.
    Each class is shown in its original color.
    """
    print("\nVisualizing separated semantic classes...")
    print("Close the window to continue.")
    
    geometries = list(separated_classes.values())
    o3d.visualization.draw_geometries(geometries,
                                      window_name="Separated Semantic Classes",
                                      width=1024,
                                      height=768)

def visualize_instances(instances_dict, show_by_class=True):
    """
    Visualize instantiated point clouds.
    Each instance is colored with a distinct color to show separation.
    
    Parameters:
    - instances_dict: Dictionary with structure {class_name: [instance1, instance2, ...]}
    - show_by_class: If True, show instances grouped by class. If False, show all instances together.
    """
    if show_by_class:
        # Show each class separately with its instances
        for class_name, instances in instances_dict.items():
            if len(instances) == 0:
                continue
                
            print(f"\nVisualizing {class_name} instances ({len(instances)} instances)...")
            
            # Generate distinct colors for each instance
            instance_colors = generate_distinct_colors(len(instances))
            
            # Create colored point clouds for each instance
            colored_instances = []
            for i, instance in enumerate(instances):
                colored_pcd = o3d.geometry.PointCloud(instance)
                # Override colors with distinct instance colors
                instance_color = np.tile(instance_colors[i], (len(instance.points), 1))
                colored_pcd.colors = o3d.utility.Vector3dVector(instance_color)
                colored_instances.append(colored_pcd)
            
            # Visualize all instances of this class together
            o3d.visualization.draw_geometries(colored_instances,
                                            window_name=f"{class_name} - {len(instances)} Instances",
                                            width=1024,
                                            height=768)
            print("Close the window to continue to next class...")
    else:
        # Show all instances from all classes together
        print("\nVisualizing all instances from all classes...")
        print("Close the window to continue.")
        
        all_colored_instances = []
        for class_name, instances in instances_dict.items():
            if len(instances) == 0:
                continue
            
            # Generate distinct colors for each instance in this class
            instance_colors = generate_distinct_colors(len(instances))
            
            for i, instance in enumerate(instances):
                colored_pcd = o3d.geometry.PointCloud(instance)
                # Override colors with distinct instance colors
                instance_color = np.tile(instance_colors[i], (len(instance.points), 1))
                colored_pcd.colors = o3d.utility.Vector3dVector(instance_color)
                all_colored_instances.append(colored_pcd)
        
        if all_colored_instances:
            o3d.visualization.draw_geometries(all_colored_instances,
                                            window_name=f"All Instances ({len(all_colored_instances)} total)",
                                            width=1024,
                                            height=768)

def visualize_summary(instances_dict, separated_classes, original_pcd):
    """
    Create a comprehensive visualization showing:
    1. Original point cloud
    2. Separated classes
    3. Instantiated instances (by class)
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION MODE")
    print("=" * 60)
    print("You will see multiple visualization windows.")
    print("Close each window to proceed to the next visualization.")
    print("=" * 60)
    
    # 1. Show original point cloud
    visualize_original_pointcloud(original_pcd)
    
    # 2. Show separated classes
    if separated_classes:
        visualize_separated_classes(separated_classes)
    
    # 3. Show instantiated instances (by class)
    if instances_dict:
        visualize_instances(instances_dict, show_by_class=False)
        
        # Ask if user wants to see all instances separately
        print("\n" + "=" * 60)
        response = input("Would you like to see all instances from all classes separately? (y/n): ")
        if response.lower() == 'y':
            visualize_instances(instances_dict, show_by_class=True)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)

def build_models(checkpoint_paths, device, num_classes=8):
    """
    Load one or more BIMNet models (for ensembling).
    """
    models = []
    for ckpt in checkpoint_paths:
        print(f"Loading checkpoint: {ckpt}")
        model = BIMNet(num_classes=num_classes)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
    return models

def voxelize_points(points, cube_edge):
    """
    Normalize points into [0, cube_edge) and create voxel occupancy grid.
    Returns:
      vox  : np.ndarray of shape [1, D, H, W]
      grid : integer indices per point, shape [N, 3]
    """
    # Normalize to [0, 1]
    min_bounds = points.min(0)
    max_bounds = points.max(0)
    points_norm = (points - min_bounds) / (max_bounds - min_bounds + 1e-8)

    # Scale to grid
    points_grid = (points_norm * (cube_edge - 1)).astype(np.int32)
    points_grid = np.clip(points_grid, 0, cube_edge - 1)

    # Occupancy grid
    vox = np.zeros((1, cube_edge, cube_edge, cube_edge), dtype=np.float32)
    vox[0, points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]] = 1.0

    return vox, points_grid

def color_label(labels, num_classes=8):
    """
    Map integer labels [D, H, W] to RGB colors using a matplotlib colormap.
    """
    cmap = plt.get_cmap("tab20", num_classes)
    flat = labels.flatten()
    colors = cmap(flat % num_classes)[:, :3]  # RGB only
    return colors.reshape((*labels.shape, 3))

def run_bimnet_inference(pcd, models, cube_edge=128, num_classes=8, device="cuda"):
    """
    Run BIMNet on a point cloud and assign semantic colors to each point.

    Returns:
      pcd          : same PointCloud object with updated colors
      preds_volume : predicted labels volume [D, H, W] (numpy)
      points_grid  : voxel indices of each point [N, 3]
    """
    points = np.asarray(pcd.points)
    print(f"Loaded {points.shape[0]} points")

    # Voxelize
    vox, points_grid = voxelize_points(points, cube_edge)

    # To tensor [B, C, D, H, W]
    x = torch.from_numpy(vox).unsqueeze(0).to(device)

    # Ensemble inference
    with torch.no_grad():
        logits_sum = None
        for model in models:
            logits = model(x)  # [B, num_classes, D, H, W]
            logits_sum = logits if logits_sum is None else logits_sum + logits

        logits_avg = logits_sum / len(models)
        preds = logits_avg.argmax(dim=1).squeeze(0).cpu().numpy()  # [D, H, W]

    # Colorize
    colors_volume = color_label(preds, num_classes=num_classes)  # [D, H, W, 3]
    point_colors = colors_volume[
        points_grid[:, 0],
        points_grid[:, 1],
        points_grid[:, 2],
    ]
    point_labels = preds[
        points_grid[:, 0],
        points_grid[:, 1],
        points_grid[:, 2]
    ]

    # unique, counts = np.unique(point_labels, return_counts=True)
    # print("\nLabel distribution over points:")
    # for u, c in zip(unique, counts):
    #     print(f"  label {u}: {c} points")

    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    return pcd, preds, points_grid, point_labels

def ransac_fit(instances_dict, target_classes=("wall", "floor", "ceiling"),
               thresh=0.01, min_points=100, max_iter=1000):
    print("Running ransac_fit...")
    print("Instances dict classes:", list(instances_dict.keys()))
    print("Target classes:", target_classes)

    planes = {}
    for class_name, instances in instances_dict.items():
        print(f"\nClass {class_name}: {len(instances)} instances")
        if class_name not in target_classes:
            print(f"  Skipping {class_name} (not in target_classes)")
            continue

        class_planes = []
        for inst_idx, inst_pcd in enumerate(instances):
            pts = np.asarray(inst_pcd.points)
            print(f"  Instance {inst_idx}: {pts.shape[0]} points")

            if pts.shape[0] < min_points:
                print(f"    Skipping: not enough points (min_points={min_points})")
                continue

            plane = pyrsc.Plane()
            equation, inliers = plane.fit(
                pts,
                thresh=thresh,
                minPoints=min_points,
                maxIteration=max_iter,
            )
            print(f"    Fitted plane: eq={equation}, inliers={len(inliers)}")

            class_planes.append({
                "equation": equation,
                "inliers": inliers,
            })

            # recolor for visualization
            colors = np.ones_like(pts) * 0.7
            for p in inliers:
                idx = np.where(np.all(pts == p, axis=1))[0]
                if len(idx) > 0:
                    colors[idx] = np.array([1.0, 0.0, 0.0])
            inst_pcd.colors = o3d.utility.Vector3dVector(colors)

        if class_planes:
            planes[class_name] = class_planes
            print(f"  -> Stored {len(class_planes)} planes for class {class_name}")
        else:
            print(f"  -> No planes stored for class {class_name}")

    print("\nDone ransac_fit.")
    return planes


def main(
    input_file,
    output_dir="output_instances",
    checkpoint_paths=None,
    cube_edge=128,
    num_classes=8,
    device=None,
    visualize_network_output=False,
    visualize_instances_flag=False,
):
    """
    Main workflow:
      1. Load point cloud
      2. BIMNet inference → semantic colors
      3. Separate by color class
      4. DBSCAN clustering per class
      5. Save instances (PLY + JSON)
      6. Optional visualizations
    """
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
        pcd,
        models,
        cube_edge=cube_edge,
        num_classes=num_classes,
        device=device,
    )

    if visualize_network_output:
        print("\nVisualizing BIMNet semantic prediction...")
        o3d.visualization.draw_geometries([pcd])

    # Step 3: Separate by semantic class (color)
    print("\nStep 1: Separating point cloud by semantic class...")
    separated_classes = separate_by_label(pcd, point_labels)

    if not separated_classes:
        print("Warning: No classes found! Check your color mappings.")
        return None

    # Step 4: Instantiate each class using DBSCAN
    print("\nStep 2: Identifying individual instances with DBSCAN...")
    all_instances = {}

    dbscan_params = {
        'ceiling':   {'eps': 0.2, 'min_points': 200},
        'floor':     {'eps': 0.3, 'min_points': 300},
        'wall':      {'eps': 0.15, 'min_points': 100},
        'beam':      {'eps': 0.1, 'min_points': 150},
        'column':    {'eps': 0.1, 'min_points': 50},
        'window':    {'eps': 0.1, 'min_points': 50},
        'door':      {'eps': 0.3, 'min_points': 300},
        'unassigned': {'eps': 0.2, 'min_points': 150},
    }

    for class_name, class_pcd in separated_classes.items():
        params = dbscan_params.get(class_name, {'eps': 0.2, 'min_points': 100})
        instances = instantiate_with_dbscan(
            class_pcd,
            class_name,
            eps=params['eps'],
            min_points=params['min_points'],
        )
        all_instances[class_name] = instances

    planes = ransac_fit(all_instances, target_classes=("wall", "floor", "ceiling"),
                        thresh=0.05, min_points=100, max_iter=1000)
    # print("Fitted planes summary:")
    # for cname, plist in planes.items():
    #     for i, p in enumerate(plist):
    #         print(f"{cname} instance {i}: equation = {p['equation']}")

    # Step 5: Save results
    print("\nStep 3: Saving instantiated point clouds...")
    save_instances(all_instances, output_dir)

    print("\n" + "=" * 60)
    print("Instantiation complete!")
    print("=" * 60)

    # Step 6: Optional visualization of instances
    if visualize_instances_flag:
        visualize_summary(all_instances, separated_classes, pcd)

    return all_instances, separated_classes, pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BIMNet semantic segmentation + DBSCAN instance extraction"
    )
    parser.add_argument(
        "--input_file",
        help="Path to input point cloud (.ply/.pcd)",
    )
    parser.add_argument(
        "--output_dir",
        default="output_instances",
        help="Directory to save instance PLYs and summary JSON",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Path(s) to BIMNet checkpoint(s). "
             "Use multiple --checkpoint args for ensembling.",
    )
    parser.add_argument(
        "--cube_edge",
        type=int,
        default=128,
        help="Voxel grid edge length (D=H=W)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="Number of BIMNet output classes",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    parser.add_argument(
        "--vis-net",
        action="store_true",
        help="Visualize BIMNet semantic prediction point cloud",
    )
    parser.add_argument(
        "--vis-instances",
        action="store_true",
        help="Visualize DBSCAN instances summary",
    )

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
        visualize_instances_flag=args.vis_instances,
    )
# if __name__ == '__main__':
#     cube_edge = 128
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # checkpoint_paths = ["log/train_pcs_bimnets3dis/val_best.pth",
#     #                     "log/train_bimnets3dis++1000/val_best.pth"]
#     checkpoint_paths = ["log/train_bimnet10_10epochs8classes/val_best.pth"]
    
#     models = []
#     for ckpt in checkpoint_paths:
#         model = BIMNet(num_classes=8)
#         model.load_state_dict(torch.load(ckpt, map_location=device))
#         model.to(device)
#         models.append(model)
#     # model = BIMNet(num_classes=13)
#     # model.load_state_dict(torch.load("log/train_s3distest/latest.pth"))
#     # model.to('cuda')
#     # dset = PCSDataset(cube_edge=cube_edge,
#     #                   augment=False,
#     #                   split='val')
#     dset = S3DISDataset(
#                  root_path="F:/data/S3DIS",
#                  splits_path="Scan-to-BIM",
#                  split="train",
#                  cube_edge=128,
#                  augment=True)
#     ids = np.indices((cube_edge, cube_edge, cube_edge)).reshape(3, -1).T

#     # pcd = o3d.io.read_point_cloud("C:/Users/amosa/Downloads/code/scantobim/Scan-to-BIM/PLY files/55L4 recre room.ply")
#     pcd = load_point_cloud(Path("C:/Users/amosa/Downloads/code/scantobim/Scan-to-BIM/PLY files/55L4 recre room.ply"))
#     if pcd.has_colors():
#         orig_colors = np.asarray(pcd.colors).copy()
#     else:
#         orig_colors = np.ones((len(pcd.points), 3)) * 0.5  # gray fallback

#     voxel_size = 0.05  # adjust this
#     points = np.asarray(pcd.points)
#     print(f"Loaded {points.shape[0]} points")

#     # === NORMALIZE AND VOXELIZE ===
#     # Normalize points to [0, cube_edge)
#     min_bounds = points.min(0)
#     max_bounds = points.max(0)
#     points_norm = (points - min_bounds) / (max_bounds - min_bounds + 1e-8)
#     points_grid = (points_norm * (cube_edge - 1)).astype(int)
#     points_grid = np.clip(points_grid, 0, cube_edge - 1)

#     # Create voxel occupancy grid [C, D, H, W]
#     vox = np.zeros((1, cube_edge, cube_edge, cube_edge), dtype=np.float32)
#     vox[0, points_grid[:,0], points_grid[:,1], points_grid[:,2]] = 1.0

#     # Convert to torch tensor [B, C, D, H, W]
#     x = torch.tensor(vox).unsqueeze(0).to(device)

#     # === MODEL INFERENCE ===
#     with torch.no_grad():
#         logits_sum = None
#         for model in models:
#             logits = model(x)  # [B, num_classes, D, H, W]
#             if logits_sum is None:
#                 logits_sum = logits
#             else:
#                 logits_sum += logits

#         # Average logits across models
#         logits_avg = logits_sum / len(models)
#         preds = logits_avg.argmax(dim=1).squeeze(0).cpu().numpy()  # [D, H, W]

#     # === COLOR MAPPING ===
#     # Fallback color map if you don’t have dset.color_label
#     def color_label(labels, num_classes=8):
#         cmap = plt.get_cmap("tab20", num_classes)
#         flat = labels.flatten()
#         colors = cmap(flat % num_classes)[:, :3]  # RGB only
#         return colors.reshape((*labels.shape, 3))

#     colors_volume = color_label(preds)  # [D, H, W, 3]

#     # === MAP COLORS BACK TO POINTS ===
#     # Find color of each point from its voxel
#     point_colors = colors_volume[
#         points_grid[:,0],
#         points_grid[:,1],
#         points_grid[:,2]
#     ]

#     # Assign colors to point cloud
#     pcd.colors = o3d.utility.Vector3dVector(point_colors)

#     # === VISUALIZE ===
#     o3d.visualization.draw_geometries([pcd])

# def main(input_file, output_dir="output_instances"):
#     """
#     Main workflow for point cloud instantiation.
#     """
#     print("=" * 60)
#     print("Point Cloud Instantiation Workflow")
#     print("=" * 60)
    
#     # Step 1: Load point cloud
#     input_path = Path(input_file)
#     pcd = load_point_cloud(input_path)
    
#     # Step 2: Separate by semantic class (color)
#     print("\nStep 1: Separating point cloud by semantic class...")
#     separated_classes = separate_by_color_class(pcd)
    
#     if not separated_classes:
#         print("Warning: No classes found! Check your color mappings.")
#         return
    
#     # Step 3: Instantiate each class using DBSCAN
#     print("\nStep 2: Identifying individual instances with DBSCAN...")
#     all_instances = {}
    
#     # DBSCAN parameters (adjust these based on your point cloud scale)
#     # eps: distance threshold - adjust based on your building scale
#     # min_points: minimum points per cluster - adjust based on point density
#     dbscan_params = {
#         'ceiling': {'eps': 0.2, 'min_points': 200},
#         'floor': {'eps': 0.3, 'min_points': 300},
#         'wall': {'eps': 0.15, 'min_points': 100},
#         'beam': {'eps': 0.1, 'min_points': 150},
#         'column': {'eps': 0.1, 'min_points': 50},
#         'window': {'eps': 0.1, 'min_points': 50},
#         'door': {'eps': 0.3, 'min_points': 300},
#         'unassigned': {'eps': 0.2, 'min_points': 150},
#     }
    
#     for class_name, class_pcd in separated_classes.items():
#         params = dbscan_params.get(class_name, {'eps': 0.2, 'min_points': 100})
#         instances = instantiate_with_dbscan(
#             class_pcd, 
#             class_name,
#             eps=params['eps'],
#             min_points=params['min_points']
#         )
#         all_instances[class_name] = instances
    
#     # Step 4: Save results
#     print("\nStep 3: Saving instantiated point clouds...")
#     save_instances(all_instances, output_dir)
    
#     print("\n" + "=" * 60)
#     print("Instantiation complete!")
#     print("=" * 60)

#     return all_instances, separated_classes, pcd

# if __name__ == "__main__":
#     # Example usage - replace with your actual file path
#     input_pointcloud = "your_pointcloud.ply"  # Change this to your file path
    
#     # You can also run from command line: python instantiate_pointcloud.py your_file.ply
#     import sys
#     visualize = False
    
#     if len(sys.argv) > 1:
#         input_pointcloud = sys.argv[1]
    
#     if len(sys.argv) > 2:
#         # Check if second arg is --visualize flag
#         if sys.argv[2] == "--visualize" or sys.argv[2] == "-v":
#             visualize = True
#             output_directory = "output_instances"
#         else:
#             output_directory = sys.argv[2]
#     else:
#         output_directory = "output_instances"
    
#     # Check for visualize flag in any position
#     if "--visualize" in sys.argv or "-v" in sys.argv:
#         visualize = True
    
#     # Run main workflow
#     result = main(input_pointcloud, output_directory)
    
#     # Visualize results if requested
#     if visualize and result:
#         all_instances, separated_classes, original_pcd = result
#         visualize_summary(all_instances, separated_classes, original_pcd)
#     elif result:
#         # Even if not explicitly requested, ask user if they want to visualize
#         all_instances, separated_classes, original_pcd = result
#         print("\n" + "=" * 60)
#         response = input("Would you like to visualize the results? (y/n): ")
#         if response.lower() == 'y':
#             visualize_summary(all_instances, separated_classes, original_pcd)