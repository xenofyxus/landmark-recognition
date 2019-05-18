from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import Net as initializeNet
import torch.optim as optim
import os

print("Is cuda available? ", torch.cuda.is_available())
x = torch.rand(5, 3)
print(x)
