import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)


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


class DDPBucket(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):

        super().__init__()

        self.module = module
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        self.buckets = []
        self.param_to_bucket_idx = {}
        self.buckets_pending_count = []
        current_bucket = []
        current_bucket_size = 0
        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue
            param_size = param.numel() * param.element_size()

            if current_bucket_size + param_size > self.bucket_size_bytes and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0

            current_bucket.append(param)
            current_bucket_size += param_size
            self.param_to_bucket_idx[param] = len(self.buckets)

        if current_bucket:
            self.buckets.append(current_bucket)

        self.buckets_pending_count = [len(bucket) for bucket in self.buckets]

        self.bucket_flat_grads = []
        for bucket in self.buckets:
            total_numel = sum(p.numel() for p in bucket)
            dtype = bucket[0].dtype
            device = bucket[0].device
            self.bucket_flat_grads.append(torch.zeros(
                total_numel, dtype=dtype, device=device))

        self.bucket_handles = [None] * len(self.buckets)

        self._register_hooks()

    def _register_hooks(self):
        def make_hook(param):
            def hook(grad):
                bucket_idx = self.param_to_bucket_idx[param]
                self.buckets_pending_count[bucket_idx] -= 1

                if self.buckets_pending_count[bucket_idx] == 0:
                    self._reduce_bucket(bucket_idx, grad, param)

            return hook

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(make_hook(param))

    def _reduce_bucket(self, bucket_idx, grad, triggering_param):

        bucket = self.buckets[bucket_idx]
        grads = []
        for p in bucket:
            # For the triggering param, the hook didnt return yet so the p.grad isnt updated yet - we need to manually set this here
            if p is triggering_param:
                grads.append(grad)
            else:
                grads.append(p.grad)

        grads[-1] = grad

        flat_grad = _flatten_dense_tensors(grads)
        self.bucket_flat_grads[bucket_idx].copy_(flat_grad)

        handle = dist.all_reduce(
            self.bucket_flat_grads[bucket_idx], op=dist.ReduceOp.SUM, async_op=True)

        self.bucket_handles[bucket_idx] = handle

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for bucket_idx, bucket in enumerate(self.buckets):
            if self.bucket_handles[bucket_idx] is not None:
                self.bucket_handles[bucket_idx].wait()
            grads = [p.grad for p in bucket]
            unflat = _unflatten_dense_tensors(
                self.bucket_flat_grads[bucket_idx], grads)

            for param, grad in zip(bucket, unflat):
                param.grad.copy_(grad/world_size)

        self._reset_state()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _reset_state(self):

        self.bucket_handles = [None] * len(self.buckets)
        self.buckets_pending_count = [len(bucket) for bucket in self.buckets]
