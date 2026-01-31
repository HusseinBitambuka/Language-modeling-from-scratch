def backward(output):
    """
    Reverse-mode autodiff starting from `output`.
    If `output` is non-scalar you must set `output.grad` before calling.
    """
    if output.grad is None:
        if output.data.ndim != 0:
            raise ValueError("Provide grad for non-scalar tensors when calling backward()")
        output.grad = 1.0

    if not output.requires_grad:
        return

    topo_ops = []
    visited_ops = set()

    def build(tensor):
        op = tensor._op
        if op is None or op in visited_ops:
            return
        visited_ops.add(op)
        for parent in tensor._parents:
            build(parent)
        topo_ops.append(op)

    build(output)

    for op in reversed(topo_ops):
        out_tensor = op.output
        grad_out = out_tensor.grad

        grads = op.backward(grad_out)

        if len(grads) != len(op.inputs):
            raise RuntimeError("Backward returned wrong number of gradients")
        for tensor, grad in zip(op.inputs, grads):
            if grad is None or not tensor.requires_grad:
                continue
            tensor.accumulate_grad(grad)
