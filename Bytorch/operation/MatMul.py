from ..Operation import Operation
from ..Tensor import Tensor

class MatMul(Operation):
    def forward(self):
        a, b = self.inputs
        self.save_for_backward(a.data, b.data)
        return a.data @ b.data
    def backward(self, grad_out):
        a, b = self.saved_tensors
        return grad_out @ b.T, a.T @ grad_out


def matmul(a, b):
    op = MatMul(a, b)
    out = Tensor(
        op.forward(),
        requires_grad=a.requires_grad or b.requires_grad,
        _op=op,
        _parents=(a, b)
    )
    op.output = out
    return out


    
