import torch 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



"""

This file contains the main implementation of the fatigue index or the CSI 
that will be later used in step function of the Adam Optimizer, when we override the "step" function of the optimizer  

"""



"""
This function makes sure, to exclude from calculations any parmeters that have no gradients
in both the current and previous gradients vectors. This is to consider the cases where there is
any freezing parameters that are with no gradients.
"""
def frozen_parameters_indices(prev_grad: np.ndarray,  current_grad:np.ndarray):
    frozen_parameters_indices =  []
    for i in range (len(prev_grad)):
        if (prev_grad[i]==0 and current_grad[i]==0):
            frozen_parameters_indices.append(i)
    return frozen_parameters_indices




def tensor_to_array(tensor: torch.Tensor):
    arr = tensor.clone().detach().cpu().numpy()
    return arr



"""
Calculating the distance between the gradient vectors of the Previous and Current step
to notice any conflict in the direction of the change. This is the first signal in the calculating of the Fatigue.

"""

def gradient_conflict(prev_grad: torch.Tensor,  current_grad:torch.Tensor):


    prev_grad= tensor_to_array(prev_grad)
    current_grad= tensor_to_array(current_grad)
    frozen_indices = frozen_parameters_indices(prev_grad=prev_grad, current_grad=current_grad)

    keep_indices = [i for i in range(len(prev_grad)) if i not in frozen_indices]

    
    
    filtered_prev_grad = prev_grad[keep_indices]

    filtered_current_grad = current_grad[keep_indices]

    filtered_prev_grad=filtered_prev_grad.reshape(1,-1)

    filtered_current_grad=filtered_current_grad.reshape(1,-1)


    similarity= cosine_similarity(filtered_current_grad, filtered_prev_grad)
    distance = 1-similarity

    return distance[0][0]



"""
This class will be used to track the error values and passed 
to the optimizer initialization
"""
class LossTracker():
    def __init__(self, window_size=20):
        self.history = []
    def log(self, loss_value):
        self.history.append(loss_value)
    def get_history(self):
        return self.history

"""
Here we are computing the changing of error signs throughout the values,
considering the instabillity is whenever the loss value is a positive 
and then goes to negative
or vice versa.
"""
def error_oscillation_calculate(loss_array_tracker: LossTracker):
    lossArray = loss_array_tracker.get_history()
    if len(lossArray)<1:
        return "NOT ENOUGH VALUES"
    deltas = []
    for i in range(len(lossArray)-1):
        deltas.append(lossArray[i+1]-lossArray[i])

    numberOfFlipsOfSign=0    
    for i in range(len(deltas)-1):
        if deltas[i]*deltas[i+1]<0:
            numberOfFlipsOfSign+=1
    normalized = numberOfFlipsOfSign/(len(deltas)-1)
    return normalized
