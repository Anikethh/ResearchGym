import torch
import torch.distributed as dist

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Allocate a list for tensors from each rank
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    
    # Perform all_gather operation
    dist.all_gather(tensors_gather, tensor, async_op=False)
    
    # Ensure the current rank's tensor is in the correct position
    tensors_gather[rank] = tensor
    
    # Concatenate tensors in the order of their ranks
    output = torch.cat(tensors_gather, dim=0)
    
    return output

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Allocate a list for tensors from each rank
        output = [torch.zeros_like(x) for _ in range(world_size)]
        
        # Perform all_gather operation
        dist.all_gather(output, x)
        
        # Ensure the current rank's tensor is in the correct position
        output[rank] = x
        
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        # Perform all_reduce operation on the gradients
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)