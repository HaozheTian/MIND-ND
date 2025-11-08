import os
import tyro
import torch
from utils import validate, load_g
from env import DismantleEnv
from dataclasses import dataclass

from networks.dismantle import SACPolicy


@dataclass
class Args:
    device: str='cuda:0'
    directory: str = 'graphs/real'
    ckpt_pth: str='saved/mind.ckpt'

args = tyro.cli(Args)
device = torch.device(args.device)
sac = SACPolicy(
    num_features=16,
    num_heads=4,
    num_mps=6,
).to(device)
sac.load_state_dict(torch.load(args.ckpt_pth, weights_only=True)['policy_state_dict'])

g_list = []
if args.directory == 'graphs/real':
    name = 'real_data'
    for type in ['bio', 'information', 'social', 'tech']:
        dir = f'{args.directory}/{type}'
        g_list.extend([load_g(os.path.join(dir, p), f'{type}_{os.path.splitext(os.path.basename(p))[0]}') 
                    for p in sorted(os.listdir(dir))])
else:
    name = 'custom_data'
    g_list.extend([load_g(os.path.join(args.directory, p), f'custom_{os.path.splitext(os.path.basename(p))[0]}') 
                    for p in sorted(os.listdir(args.directory))])
env = DismantleEnv(graph_data=g_list, batch_size=4, is_val=True)
validate(env, sac, save_res=name)

# nohup python -u test.py --device cuda:1 --ckpt_pth saved/mind.ckpt --directory graphs/real > test.out 2>&1 &
