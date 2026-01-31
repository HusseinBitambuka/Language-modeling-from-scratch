from ..Operation import Operation
import numpy as np

class Log(Operation):
    def forward(self):
        (a,) = self.inputs
        self.save_for_backward(a.data)
        return np.log(a.data)

    def backward(self, grad_out):
        (a,) = self.saved_tensors
        return (grad_out / a,)
