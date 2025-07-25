# Simulate DDP on Mac using GLOO (CPU-based) as mac does not support this so i am using this GLOO.
import torch
import torch.distributed as dist
import os

def ddp_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    print(f"I am GPU {rank}")
    print("Bye")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4  # Simulate 4 GPUs
    torch.multiprocessing.spawn(ddp_worker, args=(world_size,), nprocs=world_size)
