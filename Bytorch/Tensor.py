import numpy as np

class Tensor:
    """
    Tensor is a passive value object that participates in a computation graph.

    Invariants:
    - data is immutable w.r.t. computation
    -gradients are accumulated, never overwritten
    -Tensor owns no graph traversal logic
    """

    def __init__(self, data, *, requires_grad=False, _op=None, _parents=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad

        # Gradient is allocated lazily
        self.grad = None

        # Autograd metadata
        self._op = _op
        self._parents = _parents