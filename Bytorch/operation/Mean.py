from ..Operation import Operation
import numpy as np
class Mean (Operation):
    def forward(self):
        (a, ) = self.inputs
        self.save_for_backward(a.data.shape)
        return a.data.mean()
    def backward(self, grad_out):
        (shape, ) = self.saved_tensors
        return (grad_out * np.ones(shape) / np.prod(shape,), )
