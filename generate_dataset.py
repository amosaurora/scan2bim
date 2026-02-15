import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
import os
import random

# Create output folder
OUTPUT_DIR = "synthetic_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Class Mapping (7 Classes)
CLASSES = {
    'ceiling': 0,
    'floor': 1,
    'wall': 2,
    'beam': 3,
    'column': 4,
    'window': 5,
    'door': 6
}

def create_box_points(extents, translation, label_id, num_points):
    """Generates points on the surface of a box."""
    box = trimesh.creation.box(extents=extents)
    points, _ = trimesh.sample.sample_surface(box, num_points)
    points += translation
    labels = np.full((len(points),), label_id, dtype=np.int32)
    return points, labels

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
    # Random Dimensions
    width = np.random.uniform(4.0, 10.0)
    depth = np.random.uniform(4.0, 10.0)
    height = np.random.uniform(2.8, 4.5)
    
    # Point Density
    area_factor = (width * depth) / 16.0
    n_wall = int(8000 * area_factor)
    n_floor = int(15000 * area_factor)
    
    all_points = []
    all_labels = []

    # 1. BASICS: Floor, Ceiling, Walls
    p_floor, l_floor = create_plane(width, depth, CLASSES['floor'], n_floor)
    all_points.extend([p_floor]); all_labels.extend([l_floor])

    p_ceil, l_ceil = create_plane(width, depth, CLASSES['ceiling'], n_floor)
    p_ceil = transform_points(p_ceil, [0, 0, height])
    all_points.extend([p_ceil]); all_labels.extend([l_ceil])

    # Walls (Back, Front, Left, Right)
    walls_config = [
        ([0, depth, 0], 'x', width, height), # Back
        ([0, 0, 0], 'x', width, height),     # Front
        ([0, 0, 0], 'y', height, depth),     # Left
        ([width, 0, 0], 'y', height, depth)  # Right
    ]
    
    for pos, axis, w, h in walls_config:
        p, l = create_plane(w, h, CLASSES['wall'], n_wall)
        p = transform_points(p, pos, rotate_axis=axis)
        all_points.append(p); all_labels.append(l)

    # 2. COLUMNS (Randomly place 0-2 columns)
    if random.random() > 0.3:
        n_cols = random.randint(1, 2)
        for _ in range(n_cols):
            cx = np.random.uniform(0.5, width-0.5)
            cy = np.random.uniform(0.5, depth-0.5)
            # 0.4m x 0.4m column
            p_col, l_col = create_box_points([0.4, 0.4, height], [cx, cy, height/2], CLASSES['column'], 1500)
            all_points.append(p_col); all_labels.append(l_col)

    # 3. BEAMS (Randomly place beams near ceiling)
    if random.random() > 0.2:
        # Beam across width
        p_beam, l_beam = create_box_points([width, 0.5, 0.6], [width/2, depth/2, height-0.2], CLASSES['beam'], 3000)
        all_points.append(p_beam); all_labels.append(l_beam)

    # 4. DOORS & WINDOWS (Simulated as patches on walls)
    # Door on Back Wall
    if random.random() > 0.2:
        door_x = np.random.uniform(1.0, width-1.0)
        p_door, l_door = create_plane(1.0, 2.1, CLASSES['door'], 1200) # 1m x 2.1m door
        # Move slightly in front of wall to avoid z-fighting
        p_door = transform_points(p_door, [door_x, depth-0.05, 0], rotate_axis='x')
        all_points.append(p_door); all_labels.append(l_door)

    # Window on Left Wall
    if random.random() > 0.2:
        win_y = np.random.uniform(1.0, depth-1.0)
        p_win, l_win = create_plane(1.2, 1.2, CLASSES['window'], 1000) # 1.2m window
        p_win = transform_points(p_win, [0.05, win_y, 1.0], rotate_axis='y') # 1m height
        all_points.append(p_win); all_labels.append(l_win)

    # 5. Merge & Save
    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)
    
    # Add Sensor Noise
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
    print(f"Generated {filename}")

if __name__ == "__main__":
    # Generate 200 diverse rooms
    for i in range(200):
        fname = os.path.join(OUTPUT_DIR, f"synth_room_{i:03d}.ply")
        generate_random_room(fname)