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
                 root_path="Scan-to-BIM",
                 splits_path="Scan-to-BIM",
                 split="train",
                 cube_edge=128,
                 augment=True):

        self.root_path = root_path
        self.cube_edge = cube_edge
        self.augment = augment

        self.cmap = self.init_cmap()
        self.idmap = self.init_idmap()
        self.weights = self.init_weights()      # inverse point frequency
        self.cnames = list(self.idmap.keys())

        self.items = [l.strip() for l in open(path.join(splits_path, split+'.txt'), 'r')]

    def init_cmap(self):
        cmap = np.array(  [[128, 64,128], # ceiling
                           [244, 35,232], # floor
                           [ 70, 70, 70], # wall
                           [102,102,156], # beam
                           [190,153,153], # column
                           [153,153,153], # window
                           [250,170, 30], # door
                           [  0,  0,  0]], dtype=np.uint8) # unassigned
        return cmap

    def init_idmap(self):
        idmap = {0: 'ceiling',
                 1: 'floor',
                 2: 'wall',
                 3: 'beam',
                 4: 'column',
                 5: 'window',
                 6: 'door', 
                 7: 'unassigned',
        }
        idmap = {v:k for k,v in idmap.items()}
        return idmap
        
    def init_weights(self):
        # pts = np.array( [3370714, 2856755, 4919229, 318158, 375640, 478001, 974733, 650464, 791496, 88727, 1284130, 229758, 2272837] , dtype=np.int32)
        pts = np.array([217074, 28422379, 117924669, 546463, 3995718, 11357051, 4960637, 41740453], dtype=np.int32)
        weights = 1/(pts + 1e-6)
        # change the pts to match the remapped labels
        # merged_pts = np.array([pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7]+pts[8]+pts[9]+pts[10]+pts[11]+pts[12]])
        median_weight = np.median(weights)
        weights = np.clip(weights, a_min=None, a_max=median_weight * 10)
        return weights
    def __len__(self):
        return len(self.items)

    def color_label(self, lab, norm=True):
        if norm:
            return self.cmap[lab.numpy()]/255.
        else:
            return self.cmap[lab.numpy()]
        
    def remap_labels(self, labels):
        # Original S3DIS classes → new simplified labels
        remap_dict = {
            0: 0,   # ceiling 
            1: 1,   # floor
            2: 2,   # wall
            3: 3,   # beam
            4: 4,   # column
            5: 5,   # window
            6: 6,   # door
            7: 7,   # unassigned
            8: 7,   # table → unassigned
            9: 7,   # chair → unassigned
            10: 7,  # sofa → unassigned
            11: 7,  # bookcase → unassigned
            12: 7,  # board → unassigned
            13: 7   # clutter → unassigned
        }
        # Efficient remapping
        labels = np.vectorize(remap_dict.get)(labels)
        return labels

    def __getitem__(self, item):
        fname = path.join(self.root_path, self.items[item])
        
        data = PlyData.read(fname)
        xyz = np.array([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']]).T #np.array([[x,y,z] for x,y,z,_,_,_,_ in data['vertex']])
        lab = data['vertex'][['label']].astype(int)+1 #np.array([l for _,_,_,_,_,_,l in data['vertex']])
        lab = np.squeeze(np.array(lab))
        lab = self.remap_labels(lab)  

        # center & rescale PC in [-1,1]
        xyz -= xyz.mean(axis=0)
        xyz /= np.abs(xyz).max()

        if self.augment:

            # random rotation
            if np.random.random()<.5:
                r = R.from_rotvec(np.pi*(np.random.random(3,)-.5)*np.array([0.1,0.1,1])).as_matrix()
                xyz = np.einsum('jk,nj->nk',r,xyz)
                xyz /= np.abs(xyz).max()

            # random shift
            if np.random.random()<.5:
                xyz -= np.random.random((3,))*2-1.
            else:
                xyz += 1

            # random rescale & crop
            if np.random.random()<.5:
                if np.random.random()<.5:
                    xyz = np.round(xyz*(self.cube_edge//2)*np.random.random()).astype(int)
                else:
                    xyz = np.round(xyz*(self.cube_edge//2)/np.random.random()).astype(int)
            else:
                xyz = np.round(xyz*(self.cube_edge//2)).astype(int)

            valid = np.logical_and(np.all(xyz>-1, axis=1), np.all(xyz<self.cube_edge, axis=1))
        else:
            xyz += 1
            xyz = np.round(xyz*(self.cube_edge//2)).astype(int)
            valid = np.logical_and(np.all(xyz>-1, axis=1), np.all(xyz<self.cube_edge, axis=1))

        xyz = xyz[valid,:]
        lab = lab[valid]

        geom = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.float32)
        geom[tuple(xyz.T)] = 1

        labs = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.int64)
        labs[tuple(xyz.T)] = lab

        return torch.from_numpy(geom).unsqueeze(0), torch.from_numpy(labs)
