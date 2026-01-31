import numpy as np
from ..Operation import Operation
from ..Tensor import Tensor

class Sum (Operation):
    def forward(self):
        (a, ) = self.inputs
        self.save_for_backward(a.data.shape)
        return a.data.sum()
    def backward(self, grad_out):
        (shape, ) = self.saved_tensors
        return (grad_out * np.ones(shape),)
    

def sum_(a):
    op = Sum(a)
    out = Tensor(
        op.forward(),
        requires_grad=a.requires_grad,
        _op=op,
        _parents=(a,)
    )
    op.output = out
    return out
