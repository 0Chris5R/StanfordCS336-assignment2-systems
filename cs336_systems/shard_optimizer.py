import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)


class ShardOptimizer(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: any):

        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.param_to_rank = {}
        self.params = list(params)
        self.optimizer = None
        super().__init__(self.params, defaults={})

    def step(self, closure=None, **kwargs):

        # call the step of the wrapped optimizers and synchronize afterward
        self.optimizer.step(closure=closure, **kwargs)
        for param in self.params:
            rank = self.param_to_rank[param]
            dist.broadcast(param.data, src=rank)

    def add_param_group(self, param_group: dict[str, any]):
        params = param_group["params"]

        local_params = []

        for i, param in enumerate(params):

            rank = i % self.world_size
            self.param_to_rank[param] = rank

            if rank == self.rank:
                local_params.append(param)

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(local_params, **self.kwargs)

        else:
            self.optimizer.add_param_group({"params": local_params})

        super().add_param_group(param_group)
