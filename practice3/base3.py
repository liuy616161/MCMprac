import numpy as np
import pandas as pd
import torch
from matplotlib.pyplot import *
x=torch.linspace(0,10,200)
x.requires_grad_(True)
y=torch.sin(x)
y.backward(torch.ones_like(x))
y=y.detach()

x_array=x.detach().numpy()
y1_array=y.numpy()
y2_array=x.grad.numpy()

rc('font',size=16)
ax = subplot(1, 1, 1)
ax.plot(x_array, y1_array, 'b', label='sinx')
ax.plot(x_array, y2_array, 'r', label='cosx')
legend()
savefig('figure.png',dpi=500)
show()


