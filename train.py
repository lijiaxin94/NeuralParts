import torch
from dataloader.dataloader import build_dataloader
from model import Model
from loss.loss_function import loss_function
from metric.metric_function import metric_function
import os, sys
from config import *

def main():
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
    model = Model(device).to(device)
    train_dataloader = build_dataloader('dfaust',['train'])
    val_dataloader = build_dataloader('dfaust',['val'])
    loss_fn = loss_function
    metric_fn = metric_function
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(n_epoch):
        aggregate = 0
        loss = torch.tensor(0.0).to(device)
        sum_loss = [0.,0.,0.,0.,0.]
        sum_metric = [0.,0.]
        for b, target in zip(list(range(n_step_per_epoch)), train_dataloader.infinite_iterator()):
            aggregate += 1
            for i in range(len(target)):
                target[i] = target[i].to(device)
            prediction = model(target)
            loss = loss + loss_fn(prediction, target, sum_loss)
            metric_fn(prediction, target, sum_metric)
            #print("epoch %d, batch: %d, losses: %.5f, %.5f, %.5f, %.5f, %.5f, iou: %.5f, chamferL1: %.5f" % (epoch+1, b+1, sum_loss[0]/(b+1), sum_loss[1]/(b+1), sum_loss[2]/(b+1), sum_loss[3]/(b+1), sum_loss[4]/(b+1), sum_metric[0]/(b+1), sum_metric[1]/(b+1)))
            sys.stdout.write("epoch %d, batch: %d, losses: %.5f, %.5f, %.5f, %.5f, %.5f, iou: %.5f, chamferL1: %.5f \r" % (epoch+1, b+1, sum_loss[0]/(b+1), sum_loss[1]/(b+1), sum_loss[2]/(b+1), sum_loss[3]/(b+1), sum_loss[4]/(b+1), sum_metric[0]/(b+1), sum_metric[1]/(b+1)))
            if (aggregate == n_aggregate):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                aggregate = 0
                loss = torch.tensor(0.0).to(device)
        print("epoch %d, batch: %d, losses: %.5f, %.5f, %.5f, %.5f, %.5f, iou: %.5f, chamferL1: %.5f \r" % (epoch+1, b+1, sum_loss[0]/(b+1), sum_loss[1]/(b+1), sum_loss[2]/(b+1), sum_loss[3]/(b+1), sum_loss[4]/(b+1), sum_metric[0]/(b+1), sum_metric[1]/(b+1)))
        if (epoch+1) % 10 == 0:
            print("--------------validation--------------")
            sum_loss = [0.,0.,0.,0.,0.]
            sum_metric = [0.,0.]
            
            for b, target in zip(list(range(len(val_dataloader))), val_dataloader):
                for i in range(len(target)):
                    target[i] = target[i].to(device)
                prediction = model(target)
                loss_fn(prediction, target, sum_loss)
                metric_fn(prediction, target, sum_metric)
                sys.stdout.write("batch: %d, losses: %.5f, %.5f, %.5f, %.5f, %.5f, iou: %.5f, chamferL1: %.5f \r" % (b+1, sum_loss[0]/(b+1), sum_loss[1]/(b+1), sum_loss[2]/(b+1), sum_loss[3]/(b+1), sum_loss[4]/(b+1), sum_metric[0]/(b+1), sum_metric[1]/(b+1)))
            print("\n")
            print("--------------------------------------")


if __name__=='__main__':
    main()