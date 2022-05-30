from loss.loss_reconstruction import loss_reconstruction
from loss.loss_occupancy import loss_occupancy
from loss_normal_consistency import loss_normal_consistency
from loss_overlapping import loss_overlapping
from loss_coverage import loss_coverage
from config import *

# losses are reconstruction loss, occupancy loss, normal consistency loss, overlapping loss, and convergence loss

#output is batch * 222 * 5 * 3

def loss_function(prediction, target)
    functions = [loss_reconstruction, loss_occupancy, loss_normal_consistency,
                loss_overlapping, loss_convergence]
    losses = []
    for w, f in zip(loss_weight, functions):
        losses.append(w * f(prediction, target))
    return sum(losses)

