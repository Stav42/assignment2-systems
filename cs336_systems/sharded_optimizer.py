import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    """Keeps optimizer state for only a slice of the parameters on each rank.

    Every rank still holds all the weights and gradients; what gets partitioned
    is the optimizer state (for AdamW, two extra tensors per parameter). Each
    rank steps its own slice, then broadcasts the parameters it owns so all
    ranks end up identical again.

    Gradients are assumed to already be in sync across ranks -- that is DDP's job.
    """

    def __init__(self, params, optimizer_cls, **kwargs):
        # These must exist before super().__init__, which calls add_param_group.
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.owners = []
        self.optimizer = None

        super().__init__(params, kwargs)

    def add_param_group(self, param_group):
        # Register the whole group with the base class first: its param_groups is
        # what the inherited zero_grad() walks, so it needs every parameter.
        super().add_param_group(param_group)
        group = self.param_groups[-1]

        local = []
        for p in group["params"]:
            # Round-robin, continuing across groups rather than restarting.
            # Deterministic, so every rank derives the same assignment without talking.
            owner = len(self.owners) % self.world_size
            self.owners.append((p, owner))
            if owner == self.rank:
                local.append(p)

        if not local:
            return
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(local, **self.kwargs)
        else:
            self.optimizer.add_param_group({"params": local})

    def step(self, closure=None):
        loss = None
        if self.optimizer is not None:
            loss = self.optimizer.step(closure)

        # Each rank only updated the parameters it owns, so push those out. Every
        # rank walks the same list in the same order, so the broadcasts line up.
        for param, owner in self.owners:
            dist.broadcast(param.data, src=owner)

        return loss
