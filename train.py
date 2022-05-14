import torch
from dataloader.dataloader import build_dataloader

def main():
    dataloader = build_dataloader('dfaust',['train'])

    for b, sample in zip(list(range(3)), dataloader.infinite_iterator()):
        print(sample[0].shape, sample[1].shape, sample[2].shape)

if __name__=='__main__':
    main()