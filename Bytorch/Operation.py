class Operation:
    """
    Base class for all differiable operations

    Responsiblities:

    -store input tensors
    -store any values needed for backward
    -Define a local backward rule

    """
    def __init__(self, *inputs):
        self.inputs = inputs
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        """
        Save tensors or arrays needed for backward.
         Must be called during  farward. 
         """
        self.saved_tensors = tensors

    def forward(self):
        """
        Performs forward computation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    def backward(self, grad_out):
        """
        Given gradient w.r.t. output, return gradients w.r.t. inputs.
        
        Must return a tuple of gradients aligned with self.inputs.
        """
        raise NotImplementedError
