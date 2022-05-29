import model as model_o
import torch

def main():
    M = model_o.Model_overall(512, 5, 200, 128, 4)

    n_iter = 10
    learning_rate = 0.001
    optimizer = torch.optim.SGD(M.parameters(), lr=learning_rate)

    C = torch.rand((4,3,224,224)) # Batch * RGB * size
    pred = M(C)

    return pred 

main()