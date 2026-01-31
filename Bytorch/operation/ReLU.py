from ..Operation import Operation

class ReLU(Operation):
    def forward(self):
        (a,) = self.inputs
        mask = a.data > 0
        self.save_for_backward(mask)
        return a.data * mask

    def backward(self, grad_out):
        (mask,) = self.saved_tensors
        return (grad_out * mask,)
