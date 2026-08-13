import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DDP(nn.Module):
    def __init__(self, module: nn.Module, flat: bool = False, overlap: bool = False):
        super().__init__()

        self.overlap = overlap
        self.handles = []
        self.module = module
        self.flat = flat

        if overlap:
            for p in self.module.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._grad_hook)
        # Next thing to do: Broadcast every tensor in the model's state from rank 0 to all ranks
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def _grad_hook(self, param):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        # Keep the param so we know which gradient to average once the transfer lands.
        self.handles.append((handle, param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Code to run after the backward pass is completed, but before we take
        an optimizer step.
        """

        if self.overlap:
            for handle, param in self.handles:
                handle.wait()
                param.grad /= dist.get_world_size()
            self.handles.clear()
        elif self.flat:
            grads = [p.grad for p in self.module.parameters() if p.grad is not None]
            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= dist.get_world_size()
            for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
                g.copy_(synced)
        else:

            for param in self.module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= dist.get_world_size()
        # For example: self.module.finish_gradient_synchronization()
        