import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
from sklearn.neighbors import NearestNeighbors

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
from sklearn.decomposition import PCA
import json
from pathlib import Path
from matplotlib.colors import to_rgb
import argparse

ID_TO_NAME = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
}

def load_point_cloud(file_path):
    print(f"Loading point cloud from: {file_path}")
    
    if file_path.suffix in ['.ply', '.pcd']:
        pcd = o3d.io.read_point_cloud(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"Loaded {len(pcd.points)} points")
    return pcd

def separate_by_label(pcd, point_labels):
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

def smooth_labels_knn(pcd, labels, k=5, protected_classes=None):
    """
    Majority-vote label smoothing with an option to protect large planar classes.
    On real scans, aggressive smoothing can bleed wall labels into nearby clutter,
    so ceiling/floor/wall are left unchanged by default.
    """
    protected_classes = {0, 1, 2} if protected_classes is None else set(protected_classes)
    print(f"Smoothing labels with KNN (k={k}, protected={sorted(protected_classes)})...")
    points = np.asarray(pcd.points)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(points)
    _, indices = nbrs.kneighbors(points)

    neighbor_labels = labels[indices]
    new_labels = labels.copy()

    from scipy.stats import mode
    try:
        vote_result = mode(neighbor_labels, axis=1, keepdims=False)
        voted = np.asarray(vote_result[0]).reshape(-1)
        mask = ~np.isin(labels, list(protected_classes))
        new_labels[mask] = voted[mask]
    except Exception:
        for i in tqdm(range(len(labels)), desc="Voting"):
            if labels[i] in protected_classes:
                continue
            counts = np.bincount(neighbor_labels[i])
            new_labels[i] = np.argmax(counts)

    return new_labels

def instantiate_with_dbscan(pcd, class_name, eps=0.1, min_points=100):
    if len(pcd.points) == 0:
        return []
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"\nClustering {class_name} with DBSCAN...")
    clustering = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit(points)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    print(f"  Found {n_clusters} instances")
    
    instances = []
    for label_id in unique_labels:
        if label_id == -1:
            continue
        instance_mask = labels == label_id
        instance_pcd = pcd.select_by_index(np.where(instance_mask)[0])
        
        if is_valid_geometry(instance_pcd, class_name):
            instances.append(instance_pcd)
    
    return instances

def is_valid_geometry(pcd, class_name):
    """Helper to verify if a cluster actually looks like a column/beam."""
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    
    if class_name == 'column':
        return extent[2] > extent[0] and extent[2] > extent[1]
    if class_name == 'beam':
        return max(extent[0], extent[1]) > extent[2]
    return True

def filter_small_instances(instances_dict, min_points_thresholds):
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_name, instances in instances_dict.items():
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i, instance in enumerate(instances):
            filename = class_dir / f"{class_name}_instance_{i:03d}.ply"
            o3d.io.write_point_cloud(str(filename), instance)
        
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
            combined_pc += instance
    
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
            
            o3d.visualization.draw_geometries(
                colored_instances,
                window_name=f"{class_name} - {len(instances)} Instances",
                width=1024,
                height=768
            )
    else:
        print("\nVisualizing all instances from all classes...")
        all_colored_instances = []
        for class_name, instances in instances_dict.items():
            if len(instances) == 0:
                continue
            instance_colors = generate_distinct_colors(len(instances))
            for i, instance in enumerate(instances):
                colored_pcd = o3d.geometry.PointCloud(instance)
                instance_color = np.tile(instance_colors[i], (len(instance.points), 1))
                colored_pcd.colors = o3d.utility.Vector3dVector(instance_color)
                all_colored_instances.append(colored_pcd)
        
        if all_colored_instances:
            o3d.visualization.draw_geometries(
                all_colored_instances,
                window_name="All Instances",
                width=1024,
                height=768
            )

def visualize_summary(instances_dict, separated_classes, original_pcd):
    print("\n" + "=" * 60)
    print("VISUALIZATION MODE")
    print("=" * 60)
    
    if separated_classes:
        o3d.visualization.draw_geometries(
            list(separated_classes.values()),
            window_name="Semantic Classes",
            width=800,
            height=600
        )
    
    if instances_dict:
        visualize_instances(instances_dict, show_by_class=False)

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

def build_models(checkpoint_paths, device, num_classes=7):
    models = []
    for ckpt in checkpoint_paths:
        print(f"Loading checkpoint: {ckpt}")
        model = finetune_model(ckpt, device, num_old_classes=13, num_new_classes=num_classes)
        model.eval()
        models.append(model)
    return models

def voxelize_points(points, cube_edge):
    points_centered = points - points.mean(axis=0)
    points_centered[:, 2] -= points_centered[:, 2].min()
    
    ranges = points_centered.max(axis=0) - points_centered.min(axis=0)
    max_dim = ranges.max() + 1e-6
    scale_factor = 1.8 / max_dim
    
    points_norm = points_centered * scale_factor
    
    points_shifted = points_norm
    points_shifted[:, 2] -= 0.9
    points_shifted += 1.0

    points_grid = np.round(points_shifted * (cube_edge // 2)).astype(np.int32)
    points_grid = np.clip(points_grid, 0, cube_edge - 1)

    vox = np.zeros((1, cube_edge, cube_edge, cube_edge), dtype=np.float32)
    vox[0, points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]] = 1.0

    return vox, points_grid

def color_label(labels, num_classes=7):
    cmap = plt.get_cmap("tab20", num_classes)
    flat = labels.flatten()
    colors = cmap(flat % num_classes)[:, :3]
    return colors.reshape((*labels.shape, 3))

def run_bimnet_inference(pcd, models, cube_edge=96, num_classes=7, device="cuda"):
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

def instantiate_planar_iterative(pcd, class_name, dist_thresh=0.12, min_points=500, max_instances=12):
    """
    Iterative RANSAC for planar classes with stronger geometric checks for walls.
    """
    if len(pcd.points) < min_points:
        return []

    remaining_pcd = pcd
    instances = []
    print(f"\nRobust RANSAC for {class_name} (Thresh={dist_thresh})...")

    while len(remaining_pcd.points) > min_points and len(instances) < max_instances:
        points = np.asarray(remaining_pcd.points)
        if len(points) < min_points:
            break

        plane = pyrsc.Plane()
        try:
            best_eq, inliers = plane.fit(points, thresh=dist_thresh, minPoints=min_points, maxIteration=1000)
        except Exception:
            break

        if len(inliers) < min_points:
            break

        inst_pcd = remaining_pcd.select_by_index(inliers)
        if not is_valid_planar_instance(inst_pcd, class_name):
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            continue

        color = generate_distinct_colors(len(instances) + 1)[-1]
        inst_pcd.paint_uniform_color(color)
        instances.append(inst_pcd)

        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        print(f"  Found {class_name} instance {len(instances)}: {len(inliers)} points.")

    return instances

def oriented_line_from_wall_points(pts_xy):
    pca = PCA(n_components=2)
    pca.fit(pts_xy)
    direction = pca.components_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    center = np.median(pts_xy, axis=0)
    rel = pts_xy - center
    projections = rel @ direction
    p_min, p_max = np.percentile(projections, 5), np.percentile(projections, 95)
    start = center + direction * p_min
    end = center + direction * p_max
    normal = np.array([-direction[1], direction[0]])
    offsets = rel @ normal
    thickness = max(np.percentile(np.abs(offsets), 90) * 2.0, 0.05)
    return {
        'direction': direction,
        'center': center,
        'start': start,
        'end': end,
        'normal': normal,
        'thickness': float(thickness),
        'length': float(max(p_max - p_min, 0.0)),
        'offset_median': float(np.median(rel @ normal)),
    }

def is_valid_planar_instance(pcd, class_name):
    pts = np.asarray(pcd.points)
    if len(pts) < 100:
        return False

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())

    if class_name == 'wall':
        height = extent[2]
        horizontal = max(extent[0], extent[1])
        thickness = min(extent[0], extent[1])
        if height < 1.5 or horizontal < 0.5:
            return False
        if thickness > 1.0:
            return False
    elif class_name in ['floor', 'ceiling']:
        if extent[2] > max(extent[0], extent[1]):
            return False

    return True

def merge_collinear_walls(wall_instances, dist_tolerance=0.2, angle_tolerance_deg=8.0, gap_tolerance=0.6):
    """
    Merge wall fragments only when they are nearly parallel, lie on the same line,
    and overlap or almost touch along their dominant direction.
    """
    if not wall_instances:
        return []

    metas = []
    for inst in wall_instances:
        pts = np.asarray(inst.points)
        if len(pts) < 100:
            continue
        xy = pts[:, :2]
        line = oriented_line_from_wall_points(xy)
        z_min = np.percentile(pts[:, 2], 5)
        z_max = np.percentile(pts[:, 2], 95)
        metas.append({
            'pcd': inst,
            'xy': xy,
            'line': line,
            'z_min': float(z_min),
            'z_max': float(z_max),
        })

    if not metas:
        return []

    parent = list(range(len(metas)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    angle_tolerance = np.deg2rad(angle_tolerance_deg)

    for i in range(len(metas)):
        li = metas[i]['line']
        for j in range(i + 1, len(metas)):
            lj = metas[j]['line']

            cosang = np.clip(np.abs(np.dot(li['direction'], lj['direction'])), -1.0, 1.0)
            angle = np.arccos(cosang)
            if angle > angle_tolerance:
                continue

            mean_dir = li['direction']
            mean_normal = np.array([-mean_dir[1], mean_dir[0]])
            center_delta = metas[j]['line']['center'] - metas[i]['line']['center']
            lateral_dist = abs(np.dot(center_delta, mean_normal))
            if lateral_dist > max(dist_tolerance, 0.5 * (li['thickness'] + lj['thickness'])):
                continue

            ci = metas[i]['xy'].mean(axis=0)
            proj_i = (metas[i]['xy'] - ci) @ mean_dir
            proj_j = (metas[j]['xy'] - ci) @ mean_dir
            i0, i1 = np.percentile(proj_i, 5), np.percentile(proj_i, 95)
            j0, j1 = np.percentile(proj_j, 5), np.percentile(proj_j, 95)
            overlap = min(i1, j1) - max(i0, j0)
            gap = max(j0 - i1, i0 - j1, 0.0)
            if overlap < -gap_tolerance and gap > gap_tolerance:
                continue

            z_overlap = min(metas[i]['z_max'], metas[j]['z_max']) - max(metas[i]['z_min'], metas[j]['z_min'])
            if z_overlap < -0.2:
                continue

            union(i, j)

    grouped = {}
    for idx, meta in enumerate(metas):
        root = find(idx)
        grouped.setdefault(root, []).append(meta['pcd'])

    merged = []
    for members in grouped.values():
        combined = o3d.geometry.PointCloud()
        for inst in members:
            combined += inst
        merged.append(combined)

    print(f"  Merged {len(wall_instances)} wall segments into {len(merged)} walls")
    return merged

def instantiate_dominant_plane(pcd, class_name, dist_thresh=0.12):
    """Force-extract only the largest plausible floor/ceiling plane."""
    points = np.asarray(pcd.points)
    if len(points) < 100:
        return []

    plane = pyrsc.Plane()
    try:
        best_eq, inliers = plane.fit(points, thresh=dist_thresh, minPoints=100, maxIteration=1000)
    except Exception:
        return []

    if len(inliers) < 100:
        return []

    inst_pcd = pcd.select_by_index(inliers)
    return [inst_pcd] if is_valid_planar_instance(inst_pcd, class_name) else []

def extract_bim_parameters(instances_dict):
    """
    Robust parameter extraction using percentiles and shared room height.
    Fixes overwriting of OBB-derived object geometry for beam/column/door/window.
    """
    bim_data = []

    global_floor_z = 0.0
    global_ceiling_z = 2.5

    if 'floor' in instances_dict and len(instances_dict['floor']) > 0:
        floor_pts = np.asarray(instances_dict['floor'][0].points)
        global_floor_z = float(np.percentile(floor_pts[:, 2], 50))

    if 'ceiling' in instances_dict and len(instances_dict['ceiling']) > 0:
        ceil_pts = np.asarray(instances_dict['ceiling'][0].points)
        global_ceiling_z = float(np.percentile(ceil_pts[:, 2], 50))

    room_height = max(global_ceiling_z - global_floor_z, 0.0)

    for class_name, pcd_list in instances_dict.items():
        for idx, pcd in enumerate(pcd_list):
            pts = np.asarray(pcd.points)
            if len(pts) < 50:
                continue

            if class_name in ['beam', 'column', 'door', 'window']:
                obb = pcd.get_oriented_bounding_box()
                center = obb.center
                extent = obb.extent
                half_extent = extent / 2.0
                start = center - half_extent
                end = center + half_extent
                bim_obj = {
                    'id': f'{class_name}_{idx}',
                    'type': class_name,
                    'height': float(extent[2]),
                    'thickness': float(min(extent[0], extent[1])),
                    'geometry': {
                        'start_x': float(start[0]),
                        'start_y': float(start[1]),
                        'start_z': float(start[2]),
                        'end_x': float(end[0]),
                        'end_y': float(end[1]),
                        'end_z': float(end[2])
                    }
                }
                bim_data.append(bim_obj)
                continue

            q_min = np.percentile(pts, 5, axis=0)
            q_max = np.percentile(pts, 95, axis=0)

            bim_obj = {
                'id': f'{class_name}_{idx}',
                'type': class_name,
                'height': float(q_max[2] - q_min[2]),
                'thickness': 0.2,
                'geometry': {
                    'start_x': float(q_min[0]),
                    'start_y': float(q_min[1]),
                    'start_z': float(q_min[2]),
                    'end_x': float(q_max[0]),
                    'end_y': float(q_max[1]),
                    'end_z': float(q_min[2]),
                }
            }

            if class_name == 'floor':
                bim_obj['geometry']['start_z'] = global_floor_z
                bim_obj['geometry']['end_z'] = global_floor_z
            elif class_name == 'ceiling':
                bim_obj['geometry']['start_z'] = global_ceiling_z
                bim_obj['geometry']['end_z'] = global_ceiling_z
            elif class_name == 'wall':
                xy_pts = pts[:, :2]
                line = oriented_line_from_wall_points(xy_pts)
                bim_obj['height'] = float(room_height)
                bim_obj['thickness'] = float(min(max(line['thickness'], 0.08), 0.5))
                bim_obj['geometry']['start_z'] = global_floor_z
                bim_obj['geometry']['end_z'] = global_floor_z
                bim_obj['geometry']['start_x'] = float(line['start'][0])
                bim_obj['geometry']['start_y'] = float(line['start'][1])
                bim_obj['geometry']['end_x'] = float(line['end'][0])
                bim_obj['geometry']['end_y'] = float(line['end'][1])

            bim_data.append(bim_obj)

    return bim_data

def main(
    input_file,
    output_dir="output_instances",
    checkpoint_paths=None,
    cube_edge=96,
    num_classes=7,
    device=None,
    visualize_network_output=False,
    visualize_instances_flag=False
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = checkpoint_paths 
    
    print("=" * 60)
    print("Point Cloud Instantiation Workflow (BIMNet + DBSCAN)")
    print("=" * 60)

    input_path = Path(input_file)
    pcd = load_point_cloud(input_path)

    print("\nLoading BIMNet models...")
    models = build_models(checkpoint_paths, device, num_classes=num_classes)

    pcd, preds_volume, points_grid, point_labels = run_bimnet_inference(
        pcd, models, cube_edge=cube_edge, num_classes=num_classes, device=device
    )

    print("\nStep 0.5: Smoothing predictions with KNN...")
    point_labels = smooth_labels_knn(pcd, point_labels, k=5)
    
    print("\nStep 1: Separating point cloud by semantic class...")
    separated_classes = separate_by_label(pcd, point_labels)

    if not separated_classes:
        print("Warning: No classes found! Check your color mappings.")
        return None

    print("\nStep 2: Instantiating classes...")
    all_instances = {}

    dbscan_params = {
        'beam':   {'eps': 0.35, 'min_points': 100},
        'column': {'eps': 0.4,  'min_points': 200},
        'window': {'eps': 0.15, 'min_points': 50},
        'door':   {'eps': 0.25, 'min_points': 150},
    }

    for class_name, class_pcd in separated_classes.items():
        if class_name in ['floor', 'ceiling']:
            instances = instantiate_dominant_plane(class_pcd, class_name)
        elif class_name == 'wall':
            raw_segments = instantiate_planar_iterative(class_pcd, class_name, dist_thresh=0.12)
            instances = merge_collinear_walls(raw_segments)
        else:
            params = dbscan_params.get(class_name, {'eps': 0.3, 'min_points': 100})
            instances = instantiate_with_dbscan(class_pcd, class_name, **params)
            
        all_instances[class_name] = instances

    cleaning_thresholds = {
        'ceiling': 2000,
        'floor': 2000,
        'wall': 1000,
        'beam': 50,
        'column': 50,
        'window': 20,
        'door': 50,
    }

    all_instances = filter_small_instances(all_instances, cleaning_thresholds)

    print("\nStep 3: Extracting BIM Parameters and Saving...")
    save_instances(all_instances, output_dir)
    
    bim_json_data = extract_bim_parameters(all_instances)
    with open(Path(output_dir) / "bim_reconstruction_data.json", "w") as f:
        json.dump(bim_json_data, f, indent=4)
    print(f"BIM parameters saved to {output_dir}/bim_reconstruction_data.json")

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
    parser.add_argument("--num_classes", type=int, default=7, help="Number of BIMNet output classes")
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