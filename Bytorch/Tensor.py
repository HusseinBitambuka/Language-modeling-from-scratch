import numpy as np
from .Autograd import backward as autograd_backward


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

    def __repr__(self):
        data_str = np.array2string(
            self.data,
            precision=4,
            suppress_small=True,
            separator=", ",
            threshold=10,
        )
        grad_str = "None" if self.grad is None else np.array2string(
            self.grad,
            precision=4,
            suppress_small=True,
            separator=", ",
            threshold=6,
        )
        return (
            f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, "
            f"requires_grad={self.requires_grad}, data={data_str}, grad={grad_str})"
        )

    __str__ = __repr__

    # ----- operator overloads -----
    def _to_tensor(self, other):
        """Internal: ensure `other` is a Tensor."""
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False)

    def __add__(self, other):
        other_t = self._to_tensor(other)
        from .operation.Add import add
        return add(self, other_t)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other_t = self._to_tensor(other)
        from .operation.Mul import mul
        return mul(self, other_t)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other_t = self._to_tensor(other)
        from .operation.MatMul import matmul
        return matmul(self, other_t)

    def __rmatmul__(self, other):
        other_t = self._to_tensor(other)
        from .operation.MatMul import matmul
        return matmul(other_t, self)

    def accumulate_grad(self, grad):
        """Accumulate gradient in-place, allocating on first use."""
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

    def backward(self, grad=None):
        """
        Trigger backprop from this tensor.
        Optionally seed with an explicit gradient for non-scalar outputs.
        """
        if grad is not None:
            self.grad = np.asarray(grad, dtype=np.float32)
        autograd_backward(self)
