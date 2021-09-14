import torch
import torch.nn.modules.loss
import torch.nn.functional as F

def loss_function(adj_preds, adj_labels,weight=None):
    cost = 0.
    if weight is None:
        cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)
    else:
        cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels,weight = weight)
    
    return cost