from __future__ import print_function
import torch

print("Is cuda available? ", torch.cuda.is_available())
x = torch.rand(5, 3)
print(x)
