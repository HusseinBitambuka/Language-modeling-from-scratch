
from ..Operation import Operation
from ..Tensor import Tensor

class Add(Operation):
    def forward(self):
        a, b = self.inputs
        return a.data + b.data
    def backward(self, grad_out):
        return grad_out, grad_out
    

def add(a, b):
    op = Add(a, b)
    out = Tensor(
        op.forward(),
        requires_grad=a.requires_grad or b.requires_grad,
        _op=op,
        _parents=(a, b)
    )
    op.output = out
    return out
