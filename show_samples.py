import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.segcloud import SegCloud
from model.bimnet import BIMNet
from dataloaders.PCSdataset import PCSDataset
from dataloaders.S3DISdataset import S3DISDataset

if __name__ == '__main__':

    cube_edge = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # checkpoint_paths = ["log/train_pcs_bimnets3dis/val_best.pth",
    #                     "log/train_bimnets3dis++1000/val_best.pth"]
    checkpoint_paths = ["log/train_bimnet10_10epochs8classes/val_best.pth"]
    
    models = []
    for ckpt in checkpoint_paths:
        model = BIMNet(num_classes=8)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        models.append(model)
    # model = BIMNet(num_classes=13)
    # model.load_state_dict(torch.load("log/train_s3distest/latest.pth"))
    # model.to('cuda')
    # dset = PCSDataset(cube_edge=cube_edge,
    #                   augment=False,
    #                   split='val')
    dset = S3DISDataset(
                 root_path="F:/data/S3DIS",
                 splits_path="Scan-to-BIM",
                 split="train",
                 cube_edge=128,
                 augment=True)
    ids = np.indices((cube_edge, cube_edge, cube_edge)).reshape(3, -1).T

    # with torch.no_grad():
    #     for x, y in dset:
    #         y = y[:cube_edge, :cube_edge, :cube_edge]
    #         y -= 1
    #         cy = dset.color_label(y).reshape(-1, 3)
    #         my = y.flatten()>=0
    #         x = x.to(device).unsqueeze(0)
    #         p = model(x).argmax(dim=1).squeeze(0).cpu()
    #         cp = dset.color_label(p).reshape(-1, 3)
            
    #         ypcd = o3d.geometry.PointCloud()
    #         ypcd.points = o3d.utility.Vector3dVector(ids[my])
    #         ypcd.colors = o3d.utility.Vector3dVector(cy[my,:3])
    #         ppcd = o3d.geometry.PointCloud()
    #         ppcd.points = o3d.utility.Vector3dVector(ids[my])
    #         ppcd.colors = o3d.utility.Vector3dVector(cp[my,:3])
            
    #         vis = o3d.visualization.Visualizer()
    #         vis.create_window(width=1280, height=640)
    #         vis.add_geometry(ypcd)
    #         vis.run()
            
    #         vis = o3d.visualization.Visualizer()
    #         vis.create_window(width=1280, height=640)
    #         vis.add_geometry(ppcd)
    #         vis.run()
            
    #         vis.destroy_window()
    pcd = o3d.io.read_point_cloud("C:/Users/amosa/Downloads/code/scantobim/Scan-to-BIM/PLY files/55L4 recre room.ply")
    if pcd.has_colors():
        orig_colors = np.asarray(pcd.colors).copy()
    else:
        orig_colors = np.ones((len(pcd.points), 3)) * 0.5  # gray fallback

    voxel_size = 0.05  # adjust this
    points = np.asarray(pcd.points)
    print(f"Loaded {points.shape[0]} points")

    # === NORMALIZE AND VOXELIZE ===
    # Normalize points to [0, cube_edge)
    min_bounds = points.min(0)
    max_bounds = points.max(0)
    points_norm = (points - min_bounds) / (max_bounds - min_bounds + 1e-8)
    points_grid = (points_norm * (cube_edge - 1)).astype(int)
    points_grid = np.clip(points_grid, 0, cube_edge - 1)

    # Create voxel occupancy grid [C, D, H, W]
    vox = np.zeros((1, cube_edge, cube_edge, cube_edge), dtype=np.float32)
    vox[0, points_grid[:,0], points_grid[:,1], points_grid[:,2]] = 1.0

    # Convert to torch tensor [B, C, D, H, W]
    x = torch.tensor(vox).unsqueeze(0).to(device)

    # === MODEL INFERENCE ===
    with torch.no_grad():
        logits_sum = None
        for model in models:
            logits = model(x)  # [B, num_classes, D, H, W]
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits

        # Average logits across models
        logits_avg = logits_sum / len(models)
        preds = logits_avg.argmax(dim=1).squeeze(0).cpu().numpy()  # [D, H, W]

    # === COLOR MAPPING ===
    # Fallback color map if you don’t have dset.color_label
    def color_label(labels, num_classes=13):
        cmap = plt.get_cmap("tab20", num_classes)
        flat = labels.flatten()
        colors = cmap(flat % num_classes)[:, :3]  # RGB only
        return colors.reshape((*labels.shape, 3))

    colors_volume = color_label(preds)  # [D, H, W, 3]

    # === MAP COLORS BACK TO POINTS ===
    # Find color of each point from its voxel
    point_colors = colors_volume[
        points_grid[:,0],
        points_grid[:,1],
        points_grid[:,2]
    ]

    # Assign colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # === VISUALIZE ===
    o3d.visualization.draw_geometries([pcd])