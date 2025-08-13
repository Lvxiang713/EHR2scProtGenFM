# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 20:08:47 2025

@author: LvXiang
"""

# utils/train_diffusion.py

import os
import time
import torch
import torch.distributed as dist

def train_diffusion(
    diffusion,          # GaussianDiffusion instance
    diffusion_model,    # SingleCellAN model (possibly wrapped by DDP)
    clip_model,         # frozen CLIP model for encoding EHR
    train_loader,       # training DataLoader
    val_loader,         # validation DataLoader
    optimizer,          # optimizer (e.g. Adam)
    scheduler,          # learning‐rate scheduler (ReduceLROnPlateau)
    device,             # torch.device for computation
    num_epochs=1000,    # maximum number of epochs
    early_stop=10,      # patience for early stopping
    ckpt_dir="./diffusion_ckpt"  # directory for saving checkpoints
):
    

    """
    Train the diffusion model with optional distributed data parallel support.

    Args:
        diffusion:          GaussianDiffusion object implementing the forward/backward noising.
        diffusion_model:    The U-Net–style SingleCellAN network.
        clip_model:         A frozen CLIP model used to produce text embeddings as conditioning.
        train_loader:       DataLoader yielding (sc_data, ehr_data, _, _) for training.
        val_loader:         DataLoader yielding validation batches.
        optimizer:          Optimizer for updating model weights.
        scheduler:          Learning‐rate scheduler to step on validation loss.
        device:             Device to place tensors on.
        num_epochs:         Max epochs to train before stopping.
        early_stop:         Number of epochs with no improvement before early stopping.
        ckpt_dir:           Where to save the best model checkpoint.

    Returns:
        diffusion_model:    The model loaded with the best validation weights.
    """
    
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    # Global training set size, used for final averaging
    global_train_size = len(train_loader.dataset)
    global_val_size   = len(val_loader.dataset)
   
    for epoch in range(1, num_epochs+1):
        # ---- 1) Update DistributedSampler every epoch ----
        if dist.is_initialized() and isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
   
        # ---- 2) Training ----
        diffusion_model.train()
        local_train_sum = 0.0
        for sc_data, ehr_data, _, donor_ids in train_loader:
            sc_data  = sc_data.to(device)
            ehr_data = ehr_data.to(device)
            optimizer.zero_grad()
   
            # (1) Use frozen CLIP text encoder as condition
            with torch.no_grad():
                txt_emb = clip_model.encode_text(ehr_data)
                txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)
            cond = (
                txt_emb.unsqueeze(1),
                torch.ones(txt_emb.size(0), 1, dtype=torch.bool, device=device)
            )
            B = sc_data.size(0)

            # (2) Forward, backward, and accumulate
            loss = diffusion.train_losses(diffusion_model, sc_data, cd=cond)
            loss.backward()
            optimizer.step()
            local_train_sum += loss.item() * B
   
        # ---- 3) Aggregate training loss sum across GPUs ----
        loss_tensor = torch.tensor(local_train_sum, device=device)
        if dist.is_initialized():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_train_sum = loss_tensor.item()
        train_loss = global_train_sum / global_train_size
   
        # ---- 4) Validation ----
        diffusion_model.eval()
        local_val_sum = 0.0
        with torch.no_grad():
            for sc_data, ehr_data, _, donor_ids in val_loader:
                sc_data  = sc_data.to(device)
                ehr_data = ehr_data.to(device)
                txt_emb = clip_model.encode_text(ehr_data)
                txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)
                cond = (
                    txt_emb.unsqueeze(1),
                    torch.ones(txt_emb.size(0), 1, dtype=torch.bool, device=device)
                )
                B = sc_data.size(0)
                loss = diffusion.train_losses(diffusion_model, sc_data, cd=cond)
                local_val_sum += loss.item() * B
   
        val_tensor = torch.tensor(local_val_sum, device=device)
        if dist.is_initialized():
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        global_val_sum = val_tensor.item()
        val_loss = global_val_sum / global_val_size
   
        # ---- 5) Print and save only on rank 0 ----
        if not dist.is_initialized() or dist.get_rank() == 0:
            date_str = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())
            print(date_str)
            print(f"Epoch {epoch}/{num_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
   
        # Early stopping & checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if not dist.is_initialized() or dist.get_rank() == 0:
                torch.save(diffusion_model.state_dict(),
                           os.path.join(ckpt_dir, "best_diff_model.pth"))
                print("  Validation improved, saved model.")
        else:
            epochs_no_improve += 1
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"  No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= early_stop:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("  Early stopping.")
                break
   
        scheduler.step(val_loss)
        
    # ---- 6) Barrier to ensure checkpoint writing, then load best model on all ranks ----
    if dist.is_initialized():
        dist.barrier()
    diffusion_model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "best_diff_model.pth"), map_location=device, weights_only=True)
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Loaded best diffusion model.")
    return diffusion_model