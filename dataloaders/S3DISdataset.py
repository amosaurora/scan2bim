# import numpy as np
# import pandas as pd
# from scipy import stats
# from scipy.spatial.transform import Rotation as R
# import torch
# from torch.utils.data import Dataset
# from os import path
# from plyfile import PlyData

# class S3DISDataset(Dataset):
#     def __init__(self,
#                  root_path="../Scan-to-BIM",
#                  splits_path="../Scan-to-BIM",
#                  split="train",
#                  cube_edge=96,
#                  augment=True):

#         self.root_path = root_path
#         self.cube_edge = cube_edge
#         self.augment = augment

#         self.cmap = self.init_cmap()
#         self.idmap = self.init_idmap()
#         self.weights = self.init_weights()      # inverse point frequency
#         self.cnames = list(self.idmap.keys())

#         self.items = [l.strip() for l in open(path.join(splits_path, split+'.txt'), 'r')]

#     def init_cmap(self):
#         cmap = np.array(  [[128, 64,128], # ceiling
#                            [244, 35,232], # floor
#                            [ 70, 70, 70], # wall
#                            [102,102,156], # beam
#                            [190,153,153], # column
#                            [153,153,153], # window
#                            [250,170, 30]], # door
#                         #    [  0,  0,  0]], dtype=np.uint8) # unassigned
#                         dtype=np.uint8)
#         return cmap

#     def init_idmap(self):
#         idmap = {0: 'ceiling',
#                  1: 'floor',
#                  2: 'wall',
#                  3: 'beam',
#                  4: 'column',
#                  5: 'window',
#                  6: 'door', 
#                 #  7: 'unassigned',
#         }
#         idmap = {v:k for k,v in idmap.items()}
#         return idmap
        
#     def init_weights(self):
#         # pts = np.array( [3370714, 2856755, 4919229, 318158, 375640, 478001, 974733, 650464, 791496, 88727, 1284130, 229758, 2272837] , dtype=np.int32)
#         pts = np.array([46378544, 74583849, 216401489, 2886463, 5405718, 12127051, 5926637], dtype=np.int32)
#         # pts = np.array([39747709, 67953014, 196984929, 546463, 3995718, 11357051, 4960637, 225335105], dtype=np.int32)
#         weights = 1/(pts + 1e-6)
#         # change the pts to match the remapped labels
#         # merged_pts = np.array([pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7]+pts[8]+pts[9]+pts[10]+pts[11]+pts[12]])
#         median_weight = np.median(weights)
#         weights = np.clip(weights, a_min=None, a_max=median_weight * 3.0)
#         weights[0] *= 0.1  # Suppress Ceiling (it's too easy)
#         weights[1] *= 5.0  # Boost Floor
#         weights[2] *= 10.0
#         # weights[2] *= 2.0  # Make Walls 2x more important
#         weights[3] *= 3.0  # Make Beams 3x more important
#         weights[4] *= 3.0  # Make Columns 3x more important
#         weights[5] *= 0.5

#         # Lower Ceiling (0) importance slightly so it doesn't dominate
#         # weights[0] *= 0.5
#         return weights
    
#     # def init_weights(self):
#     #     # TEMPORARY: Force equal weights to stop "Door" dominance
#     #     return np.ones(8, dtype=np.float32)
    
#     def __len__(self):
#         return len(self.items)

#     def color_label(self, lab, norm=True):
#         if norm:
#             return self.cmap[lab.numpy()]/255.
#         else:
#             return self.cmap[lab.numpy()]
        
#     def remap_labels(self, labels):
#         # Original S3DIS classes → new simplified labels
#         remap_dict = {
#             0: 0,   # ceiling 
#             1: 1,   # floor
#             2: 2,   # wall
#             3: 3,   # beam
#             4: 4,   # column
#             5: 5,   # window
#             6: 6,   # door
#             7: -1,   # unassigned
#             8: -1,   # table → unassigned
#             9: -1,   # chair → unassigned
#             10: -1,  # sofa → unassigned
#             11: -1,  # bookcase → unassigned
#             12: -1,  # board → unassigned
#             13: -1   # clutter → unassigned
#         }
#         # Efficient remapping
#         labels = np.vectorize(remap_dict.get)(labels)
#         return labels

#     def __getitem__(self, item):
#         fname = path.join(self.root_path, self.items[item])
        
#         data = PlyData.read(fname)
#         xyz = np.array([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']]).T #np.array([[x,y,z] for x,y,z,_,_,_,_ in data['vertex']])
#         # lab = data['vertex'][['label']].astype(int)+1 #np.array([l for _,_,_,_,_,_,l in data['vertex']])
#         # Range check, disabled due to tall rooms
#         # ranges = xyz.max(axis=0) - xyz.min(axis=0)
#         # if ranges[1] > ranges[2] * 2.0: 
#         #      # Swap Y and Z to fix Y-up data
#         #     xyz = xyz[:, [0, 2, 1]] 
#         #     xyz[:, 2] *= -1
#         # xyz = xyz.astype(np.float32)
        

#         if 'label' in data['vertex']:
#             raw_lab = data['vertex']['label']
#         elif 'scalar_label' in data['vertex']:
#             raw_lab = data['vertex']['scalar_label']
#         elif 'scalar_Classification' in data['vertex']:
#             raw_lab = data['vertex']['scalar_Classification']
#         else:
#             # Print available keys to help debug if it fails again
#             available_keys = str(data['vertex'].data.dtype.names)
#             raise KeyError(f"Label field not found in {fname}. Available keys: {available_keys}")

#         # 2. Convert to integer (safely handling floats)
#         lab = np.round(raw_lab).astype(np.int32)
#         lab = np.squeeze(np.array(lab))
#         lab = self.remap_labels(lab)  
#         lab[lab > 6] = -1

#         # center & rescale PC in [-1,1]
#         # xyz -= xyz.mean(axis=0)
#         # xyz /= (np.abs(xyz).max() + 1e-8)
#         # centroid = xyz.mean(axis=0)
#         # xyz[:, 0] -= centroid[0]
#         # xyz[:, 1] -= centroid[1]
#         # xyz /= self.cube_edge

#         xyz[:, 0] -= xyz[:, 0].mean()
#         xyz[:, 1] -= xyz[:, 1].mean()
        
#         # B. Anchor Z (Floor) to 0
#         xyz[:, 2] -= xyz[:, 2].min()
        
#         # C. Scale by Largest Dimension
#         # This guarantees the WHOLE room fits in the box, whether it's tall, wide, or deep.
#         # We map the largest dimension to fit in the [-1, 1] range (Span = 2.0).
        
#         ranges = xyz.max(axis=0) - xyz.min(axis=0)
#         max_dim = ranges.max() + 1e-6 # Avoid div/0
        
#         # Scale Factor: Map max_dim to approx 1.6 (leave 10% padding)
#         scale_factor = 1.6 / max_dim
        
#         xyz *= scale_factor 
        
#         # D. Shift Z to start at -0.8 (Bottom of grid with padding)
#         xyz[:, 2] -= 0.7
        

#         if self.augment:
#             # Random Rotation: Limit tilt to Z-axis only (architectural data is gravity-aligned)
#             # Previous code allowed X/Y tilt which confuses 'Wall' vs 'Floor'
#             if np.random.random() < 0.5:
#                 angle = np.random.random() * 2 * np.pi
#                 c, s = np.cos(angle), np.sin(angle)
#                 R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
#                 xyz = np.dot(xyz, R.T)

#             # Random Shift: Keep small
#             if np.random.random() < 0.5:
#                 xyz += (np.random.random((3,)) * 0.2 - 0.1) # Shift +/- 0.1

#             # Random Rescale: Constrain to 0.8x - 1.2x
#             # (Old code allowed 0.0x to Infinity, causing spikes)
#             if np.random.random() < 0.5:
#                 scale = 0.8 + 0.4 * np.random.random()
#                 xyz *= scale

#             if np.random.random() < 0.5:
#                 noise = np.random.normal(0, 0.01, xyz.shape) # 1cm noise
#                 xyz += noise

#         xyz = xyz + 1.0 
#         xyz = np.round(xyz * (self.cube_edge // 2)).astype(np.int32)
#         # 6. Validity Check
#         valid = np.logical_and(np.all(xyz > -1, axis=1), np.all(xyz < self.cube_edge, axis=1))
#         xyz = xyz[valid,:]
#         lab = lab[valid]

#         # 7. Fill Grid
#         geom = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.float32)
#         labs = np.full((self.cube_edge, self.cube_edge, self.cube_edge), -1, dtype=np.int64)

#         # Use simple indexing. If duplicates exist, last one wins (acceptable for segmentation)
#         # Check for empty after validity filter
#         if len(xyz) > 0:
#             geom[tuple(xyz.T)] = 1
#             labs[tuple(xyz.T)] = lab
        
#         target_classes = [1, 2, 3, 4, 5, 6] # Floor, Wall, Beam, Col, Win, Door
        
#         # Get indices of existing points
#         coords = np.argwhere(geom > 0)
        
#         for x, y, z in coords:
#             lbl = labs[x, y, z]
            
#             # Only thicken non-ceiling features
#             if lbl in target_classes:
#                 # Add neighbors (Up, Down, Left, Right, Front, Back)
#                 neighbors = [
#                     (x+1, y, z), (x-1, y, z),
#                     (x, y+1, z), (x, y-1, z),
#                     (x, y, z+1), (x, y, z-1)
#                 ]
                
#                 for nx, ny, nz in neighbors:
#                     # Check bounds
#                     if 0 <= nx < self.cube_edge and \
#                        0 <= ny < self.cube_edge and \
#                        0 <= nz < self.cube_edge:
                        
#                         # Only overwrite empty space (0 intensity)
#                         if geom[nx, ny, nz] == 0:
#                             geom[nx, ny, nz] = 1.0
#                             labs[nx, ny, nz] = lbl # Propagate label

#         # if np.random.random() < 0.01:
#         #     fname_debug = f"debug_input_{np.random.randint(999)}.ply"
#         #     print(f"DEBUG: Saving voxel grid to {fname_debug}")
            
#         #     # Get coordinates of all occupied voxels
#         #     vx, vy, vz = np.where(geom > 0)
            
#         #     # Write simple PLY
#         #     with open(fname_debug, 'w') as f:
#         #         f.write(f"ply\nformat ascii 1.0\nelement vertex {len(vx)}\n")
#         #         f.write("property float x\nproperty float y\nproperty float z\n")
#         #         f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
#         #         f.write("end_header\n")
#         #         for i in range(len(vx)):
#         #             # Color it red
#         #             f.write(f"{vx[i]} {vy[i]} {vz[i]} 255 0 0\n")

#         return torch.from_numpy(geom).unsqueeze(0), torch.from_numpy(labs)
#         # if self.augment:

#         #     # random rotation
#         #     if np.random.random()<.5:
#         #         r = R.from_rotvec(np.pi*(np.random.random(3,)-.5)*np.array([0.1,0.1,1])).as_matrix()
#         #         xyz = np.einsum('jk,nj->nk',r,xyz)
#         #         xyz /= np.abs(xyz).max()

#         #     # random shift
#         #     if np.random.random()<.5:
#         #         xyz -= np.random.random((3,))*2-1.
#         #     else:
#         #         xyz += 1

#         #     # random rescale & crop
#         #     if np.random.random()<.5:
#         #         if np.random.random()<.5:
#         #             xyz = np.round(xyz*(self.cube_edge//2)*np.random.random()).astype(int)
#         #         else:
#         #             xyz = np.round(xyz*(self.cube_edge//2)/np.random.random()).astype(int)
#         #     else:
#         #         xyz = np.round(xyz*(self.cube_edge//2)).astype(int)

#         #     valid = np.logical_and(np.all(xyz>-1, axis=1), np.all(xyz<self.cube_edge, axis=1))
#         # else:
#         #     xyz += 1
#         #     xyz = np.round(xyz*(self.cube_edge//2)).astype(int)
#         #     valid = np.logical_and(np.all(xyz>-1, axis=1), np.all(xyz<self.cube_edge, axis=1))

#         # xyz = xyz[valid,:]
#         # lab = lab[valid]

#         # geom = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.float32)
#         # geom[tuple(xyz.T)] = 1

#         # labs = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.int64)
#         # labs[tuple(xyz.T)] = lab

#         # return torch.from_numpy(geom).unsqueeze(0), torch.from_numpy(labs)
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
from os import path
from plyfile import PlyData

class S3DISDataset(Dataset):
    def __init__(self,
                 root_path="../Scan-to-BIM",
                 splits_path="../Scan-to-BIM",
                 split="train",
                 cube_edge=96,
                 augment=True):

        self.root_path = root_path
        self.cube_edge = cube_edge
        self.augment = augment

        self.cmap = self.init_cmap()
        self.idmap = self.init_idmap()
        # Initialize weights last so they can use class defs if needed
        self.weights = self.init_weights()      
        self.cnames = list(self.idmap.keys())

        self.items = [l.strip() for l in open(path.join(splits_path, split+'.txt'), 'r')]

    def init_cmap(self):
        cmap = np.array(  [[128, 64,128], # ceiling
                           [244, 35,232], # floor
                           [ 70, 70, 70], # wall
                           [102,102,156], # beam
                           [190,153,153], # column
                           [153,153,153], # window
                           [250,170, 30]], # door
                        dtype=np.uint8)
        return cmap

    def init_idmap(self):
        idmap = {0: 'ceiling',
                 1: 'floor',
                 2: 'wall',
                 3: 'beam',
                 4: 'column',
                 5: 'window',
                 6: 'door', 
        }
        idmap = {v:k for k,v in idmap.items()}
        return idmap
        
    def init_weights(self):
        # Inverse point frequency from S3DIS
        pts = np.array([46378544, 74583849, 216401489, 2886463, 5405718, 12127051, 5926637], dtype=np.int32)
        weights = 1/(pts + 1e-6)
        median_weight = np.median(weights)
        # Clip to prevent Infinity, but allow high variance
        weights = np.clip(weights, a_min=None, a_max=median_weight * 10.0) 
        # --- AGGRESSIVE REBALANCING ---
        # 0:Ceil, 1:Floor, 2:Wall, 3:Beam, 4:Col, 5:Win, 6:Door
        
        weights[0] *= 1.0   # Suppress Ceiling (Too dominant)
        weights[1] *= 5.0   # Boost Floor (Often messy)
        weights[2] *= 10.0  # Boost Wall (Critical structure)
        
        # Super-Boost Rare/Thin Classes
        weights[3] *= 10.0  # Beam 
        weights[4] *= 10.0  # Column
        weights[5] *= 10.0  # Window
        weights[6] *= 10.0  # Door
        weights = weights / weights.mean()

        return weights.astype(np.float32)
    
    def __len__(self):
        return len(self.items)

    def color_label(self, lab, norm=True):
        if norm:
            return self.cmap[lab.numpy()]/255.
        else:
            return self.cmap[lab.numpy()]
        
    def remap_labels(self, labels):
        # Original S3DIS classes -> new simplified labels
        remap_dict = {
            0: 0,   # ceiling 
            1: 1,   # floor
            2: 2,   # wall
            3: 3,   # beam
            4: 4,   # column
            5: 5,   # window
            6: 6,   # door
            7: -100,  # unassigned
            8: -100,  # table -> unassigned
            9: -100,  # chair -> unassigned
            10: -100, # sofa -> unassigned
            11: -100, # bookcase -> unassigned
            12: -100, # board -> unassigned
            13: -100  # clutter -> unassigned
        }
        # Efficient remapping
        labels = np.vectorize(remap_dict.get)(labels)
        return labels

    def __getitem__(self, item):
        fname = path.join(self.root_path, self.items[item])
        
        # 1. LOAD DATA
        data = PlyData.read(fname)
        
        # Load XYZ and force FLOAT immediately
        xyz = np.array([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']]).T
        xyz = xyz.astype(np.float32) 

        # Load Labels
        if 'label' in data['vertex']:
            raw_lab = data['vertex']['label']
        elif 'scalar_label' in data['vertex']:
            raw_lab = data['vertex']['scalar_label']
        elif 'scalar_Classification' in data['vertex']:
            raw_lab = data['vertex']['scalar_Classification']
        else:
            available_keys = str(data['vertex'].data.dtype.names)
            raise KeyError(f"Label field not found in {fname}. Available keys: {available_keys}")

        # Process Labels
        lab = np.round(raw_lab).astype(np.int32)
        lab = np.squeeze(np.array(lab))
        lab = self.remap_labels(lab)  
        lab[lab > 6] = -100 # Safety mask for clutter

        # 2. PHYSICAL SCALING (WHOLE ROOM)
        
        # A. Center X and Y
        xyz[:, 0] -= xyz[:, 0].mean()
        xyz[:, 1] -= xyz[:, 1].mean()
        
        # B. Anchor Z (Floor) to 0
        xyz[:, 2] -= xyz[:, 2].min()
        
        # C. Scale by Largest Dimension
        # Map largest dimension to approx 1.6 (leaving ~20% padding in 2.0 box)
        ranges = xyz.max(axis=0) - xyz.min(axis=0)
        max_dim = ranges.max() + 1e-6 
        
        scale_factor = 1.6 / max_dim
        xyz *= scale_factor 
        
        # D. Shift Z to start at -0.8 (Bottom of grid with padding)
        xyz[:, 2] -= 0.8
        
        # 3. AUGMENTATION (Must happen BEFORE converting to Int)
        if self.augment:
            # Random Rotation: Z-axis only
            if np.random.random() < 0.5:
                angle = np.random.random() * 2 * np.pi
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                xyz = np.dot(xyz, R.T)

            # Random Shift
            if np.random.random() < 0.5:
                xyz += (np.random.random((3,)) * 0.1 - 0.05)

            # Random Rescale
            if np.random.random() < 0.5:
                scale = 0.9 + 0.2 * np.random.random() # 0.9x to 1.1x
                xyz *= scale

            # Random Jitter
            if np.random.random() < 0.5:
                noise = np.random.normal(0, 0.01, xyz.shape) 
                xyz += noise

        # 4. VOXELIZATION (The Final Step)
        # Shift [-1, 1] -> [0, 2]
        xyz = xyz + 1.0 
        
        # Scale to Grid Size [0, 96]
        # CRITICAL: Convert to Int32 ONLY HERE at the very end.
        xyz = np.round(xyz * (self.cube_edge // 2)).astype(np.int32)
        
        # 5. VALIDITY CHECK
        valid = np.logical_and(np.all(xyz >= 0, axis=1), np.all(xyz < self.cube_edge, axis=1))
        xyz = xyz[valid,:]
        lab = lab[valid]

        # 6. FILL GRID
        # Initialize Geometry
        geom = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.float32)
        
        # CRITICAL FIX: Initialize Labels to -1 (Ignore Index)
        # This prevents the model from learning "Empty Air = Ceiling (0)"
        labs = np.full((self.cube_edge, self.cube_edge, self.cube_edge), -100, dtype=np.int64)

        if len(xyz) > 0:
            # Fill existing points
            geom[tuple(xyz.T)] = 1
            labs[tuple(xyz.T)] = lab
        
            # --- NEW: ROBUST DILATION (THICKENING) ---
            # Classes to thicken: Floor(1), Wall(2), Beam(3), Col(4), Win(5), Door(6)
            # We do NOT thicken Ceiling(0) as it is already dominant.
            target_classes = [1, 2, 3, 4, 5, 6] 
            
            # Find indices where we have valid data (excluding ceiling/clutter)
            # This is faster than looping through every single voxel
            valid_indices = np.where(np.isin(labs, target_classes))
            
            if len(valid_indices[0]) > 0:
                vx, vy, vz = valid_indices
                
                # Create neighbor offsets (6-connectivity)
                offsets = [
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1)
                ]
                
                for dx, dy, dz in offsets:
                    # Shift indices
                    nx = np.clip(vx + dx, 0, self.cube_edge-1)
                    ny = np.clip(vy + dy, 0, self.cube_edge-1)
                    nz = np.clip(vz + dz, 0, self.cube_edge-1)
                    
                    # Identify empty neighbors (Intensity == 0)
                    # We only write into empty air. We never overwrite existing geometry.
                    mask_empty = (geom[nx, ny, nz] == 0)
                    
                    if np.any(mask_empty):
                        # Apply Dilation
                        fill_x = nx[mask_empty]
                        fill_y = ny[mask_empty]
                        fill_z = nz[mask_empty]
                        
                        # Copy label from the source voxel
                        source_labels = labs[vx, vy, vz]
                        
                        geom[fill_x, fill_y, fill_z] = 1.0
                        labs[fill_x, fill_y, fill_z] = source_labels[mask_empty]

        return torch.from_numpy(geom).unsqueeze(0), torch.from_numpy(labs)