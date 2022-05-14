import torch
import dataloader.dataloader.build_dataloader

def main():
    build_dataloader('dfaust',['train'])

if __name__=='__main__':
    main()