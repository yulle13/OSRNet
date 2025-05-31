import os
import subprocess
import numpy as np
import multiprocessing as mp
import math

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.nn import Module

class DistModule(Module):
    def __init__(self, module):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def average_gradients(model):
    """ average gradients """
    world_size = get_world_size()
    if world_size < 2:
        return
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)

# def broadcast_params(model):
#     """ broadcast model parameters """
#     for p in model.state_dict().values():
#         dist.broadcast(p, 0)

def dist_init(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.gpu)
        assert torch.cuda.device_count() > args.gpu, 'insufficient CUDA devices for DDP command'
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    args.dist_backend = 'nccl'
    print('| distributed init (rank:{} local rank:{}): {}'.format(
        args.rank, args.gpu, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def gather_tensors(input_array):
    world_size = dist.get_world_size()
    ## gather shapes first
    myshape = input_array.shape
    mycount = input_array.size
    shape_tensor = torch.Tensor(np.array(myshape)).cuda()
    all_shape = [torch.Tensor(np.array(myshape)).cuda() for i in range(world_size)]
    dist.all_gather(all_shape, shape_tensor)
    ## compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)
    ## padding tensors and gather them
    output_tensors = [torch.Tensor(max_count).cuda() for i in range(world_size)]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = input_array.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).cuda()
    dist.all_gather(output_tensors, input_tensor)
    ## unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [x[:all_count[i]].reshape(all_shape[i]) for i,x in enumerate(padded_output)]
    return output

def gather_tensors_batch(input_array, part_size=10):
    # gather
    rank = dist.get_rank()
    all_features = []
    part_num = input_array.shape[0] // part_size + 1 if input_array.shape[0] % part_size != 0 else input_array.shape[0] // part_size
    for i in range(part_num):
        part_feat = input_array[i * part_size:min((i+1)*part_size, input_array.shape[0]),...]
        assert part_feat.shape[0] > 0, "rank: {}, length of part features should > 0".format(rank)
        print("rank: {}, gather part: {}/{}, length: {}".format(rank, i, part_num, len(part_feat)))
        gather_part_feat = gather_tensors(part_feat)
        all_features.append(gather_part_feat)
    print("rank: {}, gather done.".format(rank))
    #all_features = np.concatenate([np.concatenate([all_features[i][j] for i in range(part_num)], axis=0) for j in range(len(all_features[0]))], axis=0)
    all_features = [np.concatenate([all_features[i][j] for i in range(part_num)], axis=0) for j in range(len(all_features[0]))]
    return all_features

def reduce_tensors(tensor:torch.Tensor)->torch.Tensor:
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    return reduced_tensor

class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, world_size=None, rank=None):
        if world_size == None:
            world_size = dist.get_world_size()
        if rank == None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        assert len(self.dataset) >= self.world_size, '{} vs {}'.format(len(self.dataset), self.world_size)
        sub_num = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        self.beg = sub_num * self.rank
        #self.end = min(self.beg+sub_num, len(self.dataset))
        self.end = self.beg + sub_num
        self.padded_ind = list(range(len(self.dataset))) + list(range(sub_num * self.world_size - len(self.dataset)))

    def __iter__(self):
        indices = [self.padded_ind[i] for i in range(self.beg, self.end)]
        return iter(indices)

    def __len__(self):
        return self.end - self.beg

class GivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size
        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter + 1) * self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        return self.total_size


class DistributedGivenIterationSampler(Sampler):  # 用于分布式训练中按给定迭代次数进行采样
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        #return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size


