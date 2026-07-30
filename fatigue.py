import torch 
from torch.optim import Adam
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
"""
This file contains the main implementation of the fatigue index or the CSI 
that will be later used in step function of the Adam Optimizer, when we override the step function  


"""

def tensor_to_array(tensor: torch.Tensor):
    arr = tensor.clone().detach().cpu().numpy()
    return arr





def gradient_conflict(prev_grad: torch.Tensor,  current_grad:torch.Tensor):

    prev_grad= tensor_to_array(prev_grad)
    current_grad= tensor_to_array(current_grad)
    prev_grad= prev_grad.reshape(1,-1)
    current_grad= current_grad.reshape(1,-1)
    similarity= cosine_similarity(prev_grad, current_grad)
    distance = 1-similarity




    return distance





def error_oscillation_calculate():
    pass
