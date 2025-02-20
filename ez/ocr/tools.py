import math
import os
from functools import partial
from pathlib import Path
import wandb

# Types
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import omegaconf
import torch
import torchvision
import logging
import omegaconf
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
# from scipy.optimize import linear_sum_assignment
# from sklearn.metrics import adjusted_rand_score

# from ocr.dinosaur.conditioning import RandomConditioning
# from ocr.dinosaur.decoding import PatchDecoder
# from ocr.dinosaur.feature_extractors.timm import TimmFeatureExtractor
# from ocr.dinosaur.neural_networks import build_two_layer_mlp, build_mlp
# from ocr.dinosaur.neural_networks.positional_embedding import DummyPositionEmbed
# from ocr.dinosaur.neural_networks.wrappers import Sequential
# from ocr.dinosaur.perceptual_grouping import SlotAttentionGrouping

Tensor = TypeVar("torch.tensor")
NN = TypeVar("torch.nn")


# [B, D, H, W] -> [B, N, D]
img_to_slot = lambda x: x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


# [B, N, D] -> [B, D, H, W]
def slot_to_img(slot):
    B, N, D = slot.shape
    size = int(math.sqrt(N))
    return slot.reshape(B, size, size, D).permute(0, 3, 1, 2)


def get_ocr_checkpoint_path(config):
    if config.local_file:
        return Path(__file__).resolve().parents[1] / config.local_file

    entity = config.entity
    project = config.project
    run_id = config.run_id
    file_path = config.file
    ocr_dir = Path(wandb.run.dir) / "ocr_checkpoints"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    run.file(file_path).download(root=ocr_dir, replace=True)
    return ocr_dir / file_path


def get_log_prefix(config):
    prefix = ""
    if config.ocr.name == "VAE":
        if config.ocr.use_cnn_feat:
            prefix = f"{config.ocr.name}N{config.ocr.cnn_feat_size**2}"
        else:
            prefix = f"{config.ocr.name}"
    elif config.ocr.name == "SlotAttn" or config.ocr.name == "SLATE":
        prefix = f"{config.ocr.name}N{config.ocr.slotattr.num_slots}"
    elif config.ocr.name == "MoNet":
        prefix = f"{config.ocr.name}N{config.ocr.num_slots}"
    else:
        prefix = f"{config.ocr.name}"
    if hasattr(config, "pooling"):
        if config.pooling.ocr_checkpoint.run_id != "":
            prefix = "Pretrained-" + prefix
        if config.pooling.learn_aux_loss:
            prefix += f"Aux"
        if config.pooling.learn_downstream_loss:
            prefix += f"FineTune"
        prefix += f"-{config.pooling.name}"
    return prefix



# https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# Taken from https://github.com/lcswillems/torch-ac/blob/master/torch_ac/utils/dictlist.py
class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.
    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value


def preprocessing_obs(obs, device, type="image"):
    ret = torch.Tensor(obs.copy()).to(device).unsqueeze(0)
    if type == "image":
        return ret.permute(0, 3, 1, 2) / 255
    elif type == "state":
        return ret


# upload batch to working device
def to_device(batch, device):
    if type(batch) == type([]):
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
    elif type(batch) == type({}):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
    else:
        batch = batch.to(device)
    return batch


# get_item from pytorch tensor
def get_item(x):
    if len(x.shape) == 0:
        return x.item()
    else:
        return x.detach().cpu().numpy()


# reshape image for visualization
for_viz = lambda x: np.array(
    x.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0, dtype=np.uint8
)


# Taken from https://github.com/singhgautam/slate/blob/master/slate.py
def visualize(images):
    B, _, H, W = images[0].shape  # first image is observation
    viz_imgs = []
    for _img in images:
        if len(_img.shape) == 4:
            viz_imgs.append(_img)
        else:
            viz_imgs += list(torch.unbind(_img, dim=1))
    viz_imgs = torch.cat(viz_imgs, dim=-1)
    # return torch.cat(torch.unbind(viz_imgs,dim=0), dim=-2).unsqueeze(0)
    return viz_imgs


# Load model and params
def load(model, agent_training=False, resume_checkpoint=None, resume_run_path=None):
    checkpoint = None
    if resume_checkpoint is not None:
        checkpoint = torch.load(
            resume_checkpoint, map_location=next(model._module.parameters()).device
        )
    elif resume_run_path is not None:
        checkpoint = torch.load(
            wandb.restore(
                "checkpoints/model_latest.pth", run_path=resume_run_path
            ).name,
            map_location=next(model._module.parameters()).device,
        )
    else:
        model_checkpoint = Path(wandb.run.dir) / "checkpoints" / "model_latest.pth"
        if model_checkpoint.exists():
            checkpoint = torch.load(
                model_checkpoint, map_location=next(model._module.parameters()).device
            )

    if checkpoint is not None:
        step = checkpoint["step"]
        if agent_training:
            episode = checkpoint["episode"]
        else:
            epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
        model.load(checkpoint)

    else:
        step = 0
        if agent_training:
            episode = 0
        else:
            epoch = 0
            best_val_loss = 1e10

    if agent_training:
        return step, episode
    else:
        return step, epoch, best_val_loss


# Save model and params
def save(
    model,
    step=0,
    epoch=0,
    best_val_loss=1e5,
    episode=0,
    agent_training=False,
    best=False,
):
    sub_dir = "checkpoints"
    model_dir = Path(wandb.run.dir) / sub_dir
    if agent_training:
        checkpoint = {"step": step, "episode": episode}
    else:
        checkpoint = {"step": step, "epoch": epoch, "best_val_loss": best_val_loss}
    checkpoint.update(model.save())
    torch.save(checkpoint, model_dir / f"model_{step}.pth")
    wandb.save(f"{sub_dir}/model_{step}.pth")
    torch.save(checkpoint, model_dir / f"model_latest.pth")
    wandb.save(f"{sub_dir}/model_latest.pth")
    if best:
        torch.save(checkpoint, model_dir / f"model_best.pth")
        wandb.save(f"{sub_dir}/model_best.pth")


# hungarian matching
def hungarian_matching(target, input, return_diff_mat=False):
    tN, tD = target.shape
    iN, iD = input.shape
    assert tN == iN and tD == iD
    diff_mat = np.zeros((tN, iN))
    for t in range(tN):
        for i in range(iN):
            diff_mat[t, i] = torch.norm(target[t] - input[i], p=1).item()
    _, col_ind = linear_sum_assignment(diff_mat)
    if return_diff_mat:
        return torch.LongTensor(col_ind).to(target.device), diff_mat[:, col_ind]
    else:
        return torch.LongTensor(col_ind).to(target.device)


# calculate ARI
def calculate_ari(true_masks, pred_masks):
    true_masks = true_masks.flatten(2)
    pred_masks = pred_masks.flatten(2)

    true_mask_ids = get_item(torch.argmax(true_masks, dim=1))
    pred_mask_ids = get_item(torch.argmax(pred_masks, dim=1))

    aris = []
    for b in range(true_mask_ids.shape[0]):
        aris.append(adjusted_rand_score(true_mask_ids[b], pred_mask_ids[b]))

    return aris


# change img numpy array to torch Tensor
def obs_to_tensor(obs, device):
    if len(obs.shape) == 4:
        return torch.Tensor(obs.transpose(0, 3, 1, 2)).to(device) / 255.0
    else:
        return torch.Tensor(obs).to(device)

