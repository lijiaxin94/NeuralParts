from metric.metric_iou import metric_iou
from metric.metric_chamferL1 import metric_chamferL1
from config import *

# iou, chamferL1

def metric_function(prediction, target, sum_metric):
    functions = [metric_iou, metric_chamferL1]
    for f in functions:
        f(prediction, target, sum_metric)

