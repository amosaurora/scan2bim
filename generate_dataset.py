import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
import os

# Create output folder
OUTPUT_DIR = "synthetic_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Class Mapping
CLASSES = {'ceiling': 0, 'floor': 1, 'wall': 2, 'unassigned': 7}

def create_plane(width, height, label_id, num_points):
    u = np.random.uniform(0, width, num_points)
    v = np.random.uniform(0, height, num_points)
    w = np.zeros(num_points)
    points = np.stack([u, v, w], axis=1)
    labels = np.full((num_points,), label_id, dtype=np.int32)
    return points, labels

def transform_points(points, translate, rotate_axis=None):
    if rotate_axis == 'x': points = points[:, [0, 2, 1]] 
    elif rotate_axis == 'y': points = points[:, [2, 1, 0]]
    points += np.array(translate)
    return points

def generate_random_room(filename):
    # --- RANDOMIZE DIMENSIONS HERE ---
    width = np.random.uniform(3.0, 8.0)   # Room width: 3m to 8m
    depth = np.random.uniform(3.0, 8.0)   # Room depth: 3m to 8m
    height = np.random.uniform(2.5, 4.0)  # Ceiling height: 2.5m to 4m
    
    # Scale point counts based on size (maintain density)
    area_factor = (width * depth) / 16.0
    n_wall = int(10000 * area_factor)
    n_floor = int(20000 * area_factor)

    all_points = []
    all_labels = []

    # 1. Floor & Ceiling
    p_floor, l_floor = create_plane(width, depth, CLASSES['floor'], n_floor)
    all_points.append(p_floor); all_labels.append(l_floor)

    p_ceil, l_ceil = create_plane(width, depth, CLASSES['ceiling'], n_floor)
    p_ceil = transform_points(p_ceil, [0, 0, height])
    all_points.append(p_ceil); all_labels.append(l_ceil)

    # 2. Walls (Back, Front, Left, Right)
    # Back
    p_w1, l_w1 = create_plane(width, height, CLASSES['wall'], n_wall)
    p_w1 = transform_points(p_w1, [0, depth, 0], rotate_axis='x')
    all_points.append(p_w1); all_labels.append(l_w1)
    
    # Front
    p_w2, l_w2 = create_plane(width, height, CLASSES['wall'], n_wall)
    p_w2 = transform_points(p_w2, [0, 0, 0], rotate_axis='x')
    all_points.append(p_w2); all_labels.append(l_w2)

    # Left
    p_w3, l_w3 = create_plane(height, depth, CLASSES['wall'], n_wall)
    p_w3 = transform_points(p_w3, [0, 0, 0], rotate_axis='y')
    all_points.append(p_w3); all_labels.append(l_w3)

    # Right
    p_w4, l_w4 = create_plane(height, depth, CLASSES['wall'], n_wall)
    p_w4 = transform_points(p_w4, [width, 0, 0], rotate_axis='y')
    all_points.append(p_w4); all_labels.append(l_w4)

    # 3. Merge & Save
    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)
    
    # Add Noise
    points += np.random.normal(0, 0.01, points.shape)

    vertex = np.empty(len(points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'i4')
    ])
    vertex['x'] = points[:,0]; vertex['y'] = points[:,1]; vertex['z'] = points[:,2]
    vertex['label'] = labels
    vertex['red'] = 100; vertex['green'] = 100; vertex['blue'] = 100

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(filename)
    print(f"Generated {filename} ({width:.1f}x{depth:.1f}m)")

# --- MAIN LOOP ---
if __name__ == "__main__":
    for i in range(200):
        fname = os.path.join(OUTPUT_DIR, f"synth_room_{i:03d}.ply")
        generate_random_room(fname)