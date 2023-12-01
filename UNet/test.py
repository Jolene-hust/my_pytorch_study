import torch

t1 = torch.tensor([
    [1,2],
    [3,4]
])
t2 = torch.tensor([
    [5,6],
    [7,8]
])
t1 = t1.to('cuda')

print(t1.device)
print(t2.device)