import torch
x=torch.arange(4.0,requires_grad=True)
y=2*torch.dot(x,x)
y.backward()
print(x.grad)

def func(op):
    if op.sum()>4:
        op=op*torch.randn(size=(2,2))+7
    else:
        op=op*op
    return op

a=torch.randn(size=(2,2),requires_grad=True)
print(a)
d=func(a)
d.backward()
print(a.grad)
