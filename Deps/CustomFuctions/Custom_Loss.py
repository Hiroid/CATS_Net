import torch
import torch.nn as nn
import torch.nn.functional as F


def context_rebel_loss(contexts_new, contexts_old, ids_new_class, basin_thd):
    loss = 0.0
    for id_new in ids_new_class:
        above_bthd_class = (contexts_new[id_new,:] - contexts_old).std(1) < basin_thd
        loss = loss - (contexts_new[id_new,:] - contexts_old[above_bthd_class,:]).var(1).sum()
    return loss

def context_rebel_loss_gaussian(contexts_new, contexts_old, ids_new_class, basin_thd):
    loss = 0.0
    for id_new in ids_new_class:
        loss = loss + torch.exp(-(contexts_new[id_new,:] - contexts_old).var(1) / basin_thd).sum()
    return loss