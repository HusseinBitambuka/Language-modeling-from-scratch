from ..Operation import Operation
import numpy as np

class Exp(Operation):
    def forward(self):
        (a, ) = self.inputs
        out = np.exp(a.data)
        self.save_for_backward(out)
        return out
    def backward(self, grad_out):
        (out, ) = self.saved_tensors
        return (grad_out * out, )
