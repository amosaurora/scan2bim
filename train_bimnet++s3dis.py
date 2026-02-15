import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from shutil import rmtree

import torch
torch.backends.cudnn.benchmark = True
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
import argparse

from model.bimnet import BIMNet
from util.losses import ClassWiseCrossEntropyLoss, HNMCrossEntropyLoss
from dataloaders.PCSdataset import PCSDataset
from dataloaders.S3DISdataset import S3DISDataset
from util.metrics import Metrics
from util.common_util import schedule, log_pcs
import warnings
warnings.filterwarnings("ignore")

#set seed for reproducibility
seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)


###### VALIDATION
# def validate(writer, vset, vloader, epoch, model, device): #PA, PP, mIoU
#     metric = Metrics(vset.cnames[1:], device=device)
#     model.eval()
#     with torch.no_grad():
#         for x, y in tqdm(vloader, "Validating Epoch %d"%(epoch+1), total=len(vset)):
#             x, y = x.to(device), y.to(device, dtype=torch.long)-1 # shift indices 
#             o = model(x)
#             metric.add_sample(o.argmax(dim=1).flatten(), y.flatten())
#             #break
#     miou = metric.percent_mIoU()
#     acc = metric.percent_acc()
#     prec = metric.percent_prec()
#     writer.add_scalar('mIoU', miou, epoch)
#     writer.add_scalar('PP', prec, epoch)
#     writer.add_scalar('PA', acc, epoch)
#     writer.add_scalars('IoU', {n:100*v for n,v in zip(metric.name_classes, metric.IoU()) if not torch.isnan(v)}, epoch)
#     print(metric)
#     model.train()
#     return miou, o, y

def validate(writer, vset, vloader, epoch, model, device, criterion): 
    metric = Metrics(vset.cnames, device=device)
    # metric = Metrics(vset.cnames[1:], device=device)

    model.eval()
    
    val_loss = 0.0  # Initialize loss accumulator
    
    with torch.no_grad():
        for x, y in tqdm(vloader, "Validating Epoch %d"%(epoch+1), total=len(vloader)): # changed len(vset) to len(vloader) for accuracy
            x, y = x.to(device), y.to(device, dtype=torch.long) 
            o = model(x)
            
            # --- NEW: Calculate Validation Loss ---
            # We use the same loss function as training (or a standard one)
            l = criterion(o, y)
            val_loss += l.item()

            metric.add_sample(o.argmax(dim=1).flatten(), y.flatten())
            
    # Average the loss over the number of batches
    val_loss /= len(vloader)
    
    miou = metric.percent_mIoU()
    acc = metric.percent_acc()
    prec = metric.percent_prec()
    
    # --- NEW: Log Validation Loss to TensorBoard ---
    writer.add_scalar('Val/mIoU', miou, epoch)
    writer.add_scalar('Val/Loss', val_loss, epoch) # <--- Track this!
    writer.add_scalar('Val/Precision', prec, epoch)
    writer.add_scalar('Val/Accuracy', acc, epoch)
    # -----------------------------------------------
    
    print(f"\nEpoch {epoch+1} Validation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  mIoU: {miou:.2f}%")
    print(metric)
    
    model.train()
    return miou, val_loss, o, y  # Return val_loss

# def load_and_prune_weights(model_7class, checkpoint_path_8class):
#     print(f"Loading 8-class weights from {checkpoint_path_8class}...")
#     state_8 = torch.load(checkpoint_path_8class)
#     state_7 = model_7class.state_dict()
    
#     for key in state_7:
#         if key in state_8:
#             if state_7[key].shape == state_8[key].shape:
#                 state_7[key] = state_8[key]
#             else:
#                 print(f"Pruning layer: {key} | Old: {state_8[key].shape} -> New: {state_7[key].shape}")
                
#                 # CASE 1: Bias (1D Tensor) -> Slice Dim 0 only
#                 if len(state_7[key].shape) == 1:
#                     state_7[key][:7] = state_8[key][:7]
                    
#                 # CASE 2: Weights (Multi-Dim Tensor)
#                 else:
#                     # Slice Dimension 0 (Output Channels)
#                     # We create a temporary slice first
#                     temp_slice = state_8[key][:7]
                    
#                     # Check if Dimension 1 (Input Channels) also needs slicing
#                     if state_7[key].shape[1] < temp_slice.shape[1]:
#                         # Slice Dim 1 as well: [7, 8, ...] -> [7, 7, ...]
#                         state_7[key][:7, :7] = temp_slice[:, :7]
#                     else:
#                         # Only Dim 0 needed slicing
#                         state_7[key][:7] = temp_slice

#     model_7class.load_state_dict(state_7)
#     print("Surgery complete. Model ready for 7-class fine-tuning.")
#     return model_7class



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs to run')
    parser.add_argument("--batch_size", type=int, default=4, help='batch_size')
    parser.add_argument("--cube_edge", type=int, default=96, help='granularity of voxelization train')
    parser.add_argument("--val_cube_edge", type=int, default=96, help='granularity of voxelization val')
    parser.add_argument("--num_classes", type=int, default=7, help='number of classes to consider')
    parser.add_argument("--dset_path", type=str, default="/media/elena/M2SSD/datasets/HePIC/HePIC", help='dataset path')
    parser.add_argument("--test_name", type=str, default='test', help='optional test name')
    parser.add_argument("--pretrain", type=str, help='pretrained model path')
    parser.add_argument("--loss", choices=['ce','cwce','ohem','mixed'], default='mixed', type=str, help='which loss to use')
    args = parser.parse_args()

    lr0 = 5e-5
    lre = 1e-5
    eval_every_n_epochs = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logdir = "log/train_bimnet++" + "_" + args.test_name
    rmtree(logdir, ignore_errors=True)
    writer = SummaryWriter(logdir, flush_secs=.5)

    # Load model
    model = BIMNet(args.num_classes)
    if args.pretrain:
        # model = load_and_prune_weights(model, args.pretrain)
        new = model.state_dict()
        old = torch.load(args.pretrain)
        for k in new:
            if "out" not in k:
                new[k] = old[k]
        model.load_state_dict(new)
        print("model restored from ", args.pretrain)
    model.to(device)
        
    # Load dataset
    dataset = S3DISDataset
    dset = dataset(root_path=args.dset_path,
                   #fsl=50,
                   cube_edge=args.cube_edge)
    dloader = DataLoader(dset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=4,
                         drop_last=True)

    vset = dataset(root_path=args.dset_path,
                   cube_edge=args.val_cube_edge,
                   augment=False,
                   split='val')
    vloader = DataLoader(vset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)


    # set up parameters for training
    steps_per_epoch = len(dset)//args.batch_size
    tot_steps = steps_per_epoch*args.epochs
    optim = Adam(model.parameters(), weight_decay=1e-6)
    

    # to visualize point cloud
    pts = 2*torch.from_numpy(np.indices((args.val_cube_edge, args.val_cube_edge, args.val_cube_edge))
                             .reshape(3, -1).T).unsqueeze(0)/args.cube_edge - 1.
    best_miou = 0

    best_val_loss = float('inf')  # Start with infinity so first loss is always lower
    val_criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    if args.loss == 'ce':
        loss = nn.CrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'cwce':
        loss = ClassWiseCrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'ohem':
        loss = HNMCrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'mixed':
        loss1 = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.sqrt(
                    torch.tensor(dset.weights, dtype=torch.float32,
                                device=device)))  # weight=torch.tensor(dset.weights, dtype=torch.float32, device=device))
        loss2 = ClassWiseCrossEntropyLoss(ignore_index=-1, 
                    weight=torch.tensor(np.ones_like(dset.weights), dtype=torch.float32, device=device))
    else:
        raise NotImplementedError

    # TRAINING PHASE
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze ONLY the final output layer (classifier)
    for param in model.out.parameters():
        param.requires_grad = True

    for e in range(args.epochs):
        torch.cuda.empty_cache()

        #Evaluate every n epochs
        # if e % eval_every_n_epochs == 0:           
        #     if e>=0:
        #         miou, o, y = validate(writer, vset, vloader, e, model, device)
        #         if miou>best_miou:
        #             best_miou = miou
        #             torch.save(model.state_dict(), logdir+"/val_best.pth")
        #         #log_pcs(writer, dset, pts, o, y)
        #     metrics = Metrics(dset.cnames[1:], device=device)
        if e % eval_every_n_epochs == 0:           
            if e >= 0:
                # Pass val_criterion to the function
                miou, val_loss, o, y = validate(writer, vset, vloader, e, model, device, val_criterion)

                # Check 1: Best mIoU (Original logic)
                if miou > best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), logdir+"/val_best_miou.pth")
                    print(f"  Saved new best mIoU model: {best_miou:.2f}%")

                # Check 2: Best Loss (Global Minimum logic)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), logdir+"/val_best_loss.pth")
                    print(f"  Saved new best LOSS model: {best_val_loss:.4f}")

                metrics = Metrics(dset.cnames, device=device)
                # metrics = Metrics(dset.cnames[1:], device=device)

        if e == 5:
            print("Unfreezing encoder...")
            for param in model.parameters():
                param.requires_grad = True
       
        pbar = tqdm(dloader, total=steps_per_epoch, desc="Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress"%(e+1, args.epochs, 0., 0.))

        for i, (x, y) in enumerate(pbar):

            step = i+steps_per_epoch*e

            lam = schedule(0, 1, step, tot_steps, .9)
           
            lr = schedule(lr0, lre, step, tot_steps, .9)
            optim.param_groups[0]['lr'] = lr
            optim.zero_grad()
            
            x, y = x.to(device), y.to(device, dtype=torch.long) # shift indices 
            
            o = model(x)
            if args.loss == 'mixed':
                l = loss2(o, y) * (1 - lam) + loss1(o, y) * (lam)
            else:
                l = loss(o, y)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            metrics.add_sample(o.detach().argmax(dim=1).flatten(), y.flatten())

            optim.step()
            miou = metrics.percent_mIoU()
            pbar.set_description("Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress"%(e+1, args.epochs, l.item(), miou))
            
            writer.add_scalar('lr', lr, step)
            writer.add_scalar('loss', l.item(), step)
            writer.add_scalar('step_mIoU', miou, step)

        torch.save(model.state_dict(), logdir+"/latest.pth")
        
    # EVALUATION
    miou, val_loss, o, y = validate(writer, vset, vloader, e, model, device, val_criterion)
    
    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), logdir+"/val_best_miou.pth")
        
    # Optional: Save if this final run happened to be the best loss too
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), logdir+"/val_best_loss.pth")
