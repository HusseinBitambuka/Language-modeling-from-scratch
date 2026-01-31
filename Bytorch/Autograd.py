def backward(output):
    if output.data.ndim != 0:
        raise ValueError("backward () can only be called on a scalar tensor")
    if not output.requires_grad:
        return
    #seed
    output.grad = 1.0
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