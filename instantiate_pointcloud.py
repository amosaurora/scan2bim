"""
Point Cloud Instantiation Script
Separates semantic point cloud by class and identifies individual instances using DBSCAN clustering.
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


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


def separate_by_color_class(pcd):
    """
    Separate point cloud by semantic class (color).
    
    Classes (based on your BIMNet results):
    - Walls
    - Floors
    - Beams
    - Columns
    - Doors
    - Windows
    - Roofs
    - Stairs
    
    Returns: Dictionary mapping class names to point clouds
    """
    if not pcd.has_colors():
        raise ValueError("Point cloud must have color information for semantic separation")
    
    # Get points and colors as numpy arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Convert colors to integer RGB values (0-255)
    colors_int = (colors * 255).astype(np.uint8)
    
    # Define color mappings for each class
    # TODO: You'll need to adjust these based on your actual color scheme
    # Common color schemes: check what colors BIMNet uses
    class_colors = {
        'ceiling': (128, 64,128),      # Red - adjust as needed
        'floor': (244, 35,232),     # Green - adjust as needed
        'wall': (70, 70, 70),      # Blue - adjust as needed
        'beam': (102,102,156),  # Yellow - adjust as needed
        'column': (102,102,156),    # Magenta - adjust as needed
        'window': (190,153,153),  # Cyan - adjust as needed
        'door': (153,153,153),    # Purple - adjust as needed
        'unassigned': (0,  0,  0),   # Orange - adjust as needed
    }
    
    # Tolerance for color matching (colors might not be exact due to processing)
    color_tolerance = 10
    
    separated_classes = {}
    
    for class_name, target_color in class_colors.items():
        # Find points within color tolerance
        color_diff = np.abs(colors_int - np.array(target_color))
        matching_mask = np.all(color_diff <= color_tolerance, axis=1)
        
        if np.any(matching_mask):
            class_points = points[matching_mask]
            class_colors_array = colors[matching_mask]
            
            # Create point cloud for this class
            class_pcd = o3d.geometry.PointCloud()
            class_pcd.points = o3d.utility.Vector3dVector(class_points)
            class_pcd.colors = o3d.utility.Vector3dVector(class_colors_array)
            
            separated_classes[class_name] = class_pcd
            print(f"  {class_name}: {len(class_points)} points")
    
    return separated_classes


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
        visualize_instances(instances_dict, show_by_class=True)
        
        # Ask if user wants to see all instances together
        print("\n" + "=" * 60)
        response = input("Would you like to see all instances from all classes together? (y/n): ")
        if response.lower() == 'y':
            visualize_instances(instances_dict, show_by_class=False)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)

def main(input_file, output_dir="output_instances"):
    """
    Main workflow for point cloud instantiation.
    """
    print("=" * 60)
    print("Point Cloud Instantiation Workflow")
    print("=" * 60)
    
    # Step 1: Load point cloud
    input_path = Path(input_file)
    pcd = load_point_cloud(input_path)
    
    # Step 2: Separate by semantic class (color)
    print("\nStep 1: Separating point cloud by semantic class...")
    separated_classes = separate_by_color_class(pcd)
    
    if not separated_classes:
        print("Warning: No classes found! Check your color mappings.")
        return
    
    # Step 3: Instantiate each class using DBSCAN
    print("\nStep 2: Identifying individual instances with DBSCAN...")
    all_instances = {}
    
    # DBSCAN parameters (adjust these based on your point cloud scale)
    # eps: distance threshold - adjust based on your building scale
    # min_points: minimum points per cluster - adjust based on point density
    dbscan_params = {
        'ceiling': {'eps': 0.2, 'min_points': 200},
        'floor': {'eps': 0.3, 'min_points': 300},
        'wall': {'eps': 0.15, 'min_points': 100},
        'beam': {'eps': 0.1, 'min_points': 150},
        'column': {'eps': 0.1, 'min_points': 50},
        'window': {'eps': 0.1, 'min_points': 50},
        'door': {'eps': 0.3, 'min_points': 300},
        'unassigned': {'eps': 0.2, 'min_points': 150},
    }
    
    for class_name, class_pcd in separated_classes.items():
        params = dbscan_params.get(class_name, {'eps': 0.2, 'min_points': 100})
        instances = instantiate_with_dbscan(
            class_pcd, 
            class_name,
            eps=params['eps'],
            min_points=params['min_points']
        )
        all_instances[class_name] = instances
    
    # Step 4: Save results
    print("\nStep 3: Saving instantiated point clouds...")
    save_instances(all_instances, output_dir)
    
    print("\n" + "=" * 60)
    print("Instantiation complete!")
    print("=" * 60)

    return all_instances, separated_classes, pcd

if __name__ == "__main__":
    # Example usage - replace with your actual file path
    input_pointcloud = "your_pointcloud.ply"  # Change this to your file path
    
    # You can also run from command line: python instantiate_pointcloud.py your_file.ply
    import sys
    visualize = False
    
    if len(sys.argv) > 1:
        input_pointcloud = sys.argv[1]
    
    if len(sys.argv) > 2:
        # Check if second arg is --visualize flag
        if sys.argv[2] == "--visualize" or sys.argv[2] == "-v":
            visualize = True
            output_directory = "output_instances"
        else:
            output_directory = sys.argv[2]
    else:
        output_directory = "output_instances"
    
    # Check for visualize flag in any position
    if "--visualize" in sys.argv or "-v" in sys.argv:
        visualize = True
    
    # Run main workflow
    result = main(input_pointcloud, output_directory)
    
    # Visualize results if requested
    if visualize and result:
        all_instances, separated_classes, original_pcd = result
        visualize_summary(all_instances, separated_classes, original_pcd)
    elif result:
        # Even if not explicitly requested, ask user if they want to visualize
        all_instances, separated_classes, original_pcd = result
        print("\n" + "=" * 60)
        response = input("Would you like to visualize the results? (y/n): ")
        if response.lower() == 'y':
            visualize_summary(all_instances, separated_classes, original_pcd)


# python -m venv .venv
# . .\.venv\Scripts\Activate.ps1
# pip install -r requirements.txt
# python instantiate_pointcloud.py "C:\path\to\your_pointcloud.ply" "output_instances"