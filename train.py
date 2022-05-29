import torch
from dataloader.dataloader import build_dataloader
from model import Model
from get_loss_function import get_loss_function
from config import *

def main():
    device =  torch.device(cuda if torch.cuda.is_available() else cpu )
    model = Model('dfaust')
    dataloader = build_dataloader('dfaust',['train'])
    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(n_epoch):
        aggregate = 0
        for b, target in zip(list(range(n_step_per_epoch)), dataloader.infinite_iterator()):
            aggregate += 1
            for i in range(len(sample)):
                target[i] = target[i].to(device)
            prediction = model(target)
            loss = loss_fn(prediction, target)
            if (aggregate == n_aggregate)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                aggregate = 0

if __name__=='__main__':
    main()