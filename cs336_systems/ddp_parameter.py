import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class DDPParameter(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):

        super().__init__()

        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._hook_function())

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.div_(self.world_size)

    def _hook_function(self):
        def hook(param):
            handle = dist.all_reduce(
                param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        return hook
