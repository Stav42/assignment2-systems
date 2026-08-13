import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()

        self.module = module
        # Next thing to do: Broadcast every tensor in the model's state from rank 0 to all ranks
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Code to run after the backward pass is completed, but before we take
        an optimizer step.
        """

        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()
        # For example: self.module.finish_gradient_synchronization()
        