import os
import tyro
import time
import torch
import random
import numpy as np
from typing import Optional
from collections import deque
from dataclasses import dataclass
from torch_scatter import scatter_add
from datetime import datetime, timedelta
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from env import DismantleEnv
from networks.dismantle import load_dismantler
from utils import ReplayBuffer, Batch, validate, ig_to_data



@dataclass
class Args:
    use_tb: bool=False
    """record using tensorboard"""
    seed: int=0
    """random seed"""
    device: str='cuda:0'
    """the device to use"""
    num_envs: int=64
    """number of parallel environments"""
    total_steps: int=200000
    """number of training steps (transitions = steps*num_envs)"""
    buffer_size: int=2000000
    """size of the replay buffer"""
    batch_size: int=512
    """batch size for updating networks"""
    val_frequency: int=1000
    """validation frequency"""
    save_frequency: int=1000
    """save frequency"""
    learning_starts: int= 2000
    """timestep to start learning"""
    learning_rate: float=3e-4
    """learning rate for the policy and the Q networks"""
    tau: float=1.0
    """target smoothing factor"""
    alpha: float=0.005
    """intensity of entropy regularization"""
    gamma: float=0.99
    """Discount factor"""
    num_updates: int=16
    """number of network updates at each step"""
    target_frequency: int=200
    """the frequency for updating the target networks"""
    ckpt_pth: Optional[str]=None
    """where ckeckpoint was saved"""
    num_features: int = 16
    """number of initial node features"""
    num_heads: int=4
    """number of message passings heads"""
    num_mps: int=6
    """number of message passings"""
    normalize: bool=True
    """apply instance normalization"""



if __name__ == "__main__":
    args = tyro.cli(Args)
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    run_path = f"{time_string}"
    device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.use_tb:
        writer = SummaryWriter(f"runs/{run_path}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" 
                for key, value in vars(args).items()])),
        )

    print(f'Training starts at {time_string}')
    print(f'Device is {device}. Seed set to {args.seed}')

    env = DismantleEnv(
        data_dir='graphs/train', 
        batch_size=args.num_envs, 
        is_val=False, 
        seed=args.seed,
        remove_scc=False
    )
    env_val = DismantleEnv(
        data_dir='graphs/validation', 
        batch_size=args.num_envs, 
        is_val=True, 
        seed=args.seed
    )
    
    buffer = ReplayBuffer(args.buffer_size, device)

    policy, qf1, qf2, qf1_target, qf2_target = load_dismantler(args.num_features, args.num_heads, args.num_mps, device, args.ckpt_pth)
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate, eps=1e-4)
    policy_optimizer = torch.optim.Adam(list(policy.parameters()), lr=args.learning_rate, eps=1e-4)

    num_eps, num_updates = 0, 0
    auc_buffer = deque(maxlen=20)
    start_time = time.time()

    #### MAIN LOOP ####
    obs_list, _ = env.reset()
    for global_step in range(args.total_steps): # args.num_envs transitions at each global step
        
        #### DISMANTLE ####
        if global_step<args.learning_starts and args.ckpt_pth==None:
            act_arr = env.sample_act()
        else:
            with torch.no_grad():
                act_arr, _ = policy.get_action(Batch(device, [ig_to_data(g) for g in obs_list]))
            act_arr = act_arr.detach().cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)

        buffer.add(obs_list, act_arr, obs_next_list, rew_arr, done_arr)

        obs_next_list, _ = env.reset_async(done_arr)
        
        obs_list = obs_next_list

        for logger in info_list: # info_dict is empty unless some eps are finished
            num_eps += 1
            auc_buffer.append(logger.auc/logger.n_init)
        
        if global_step % args.val_frequency == 0 and global_step >= args.learning_starts:
            val_auc_list, _ = validate(env_val, policy)
            auc_val_avg = sum(val_auc_list)/len(val_auc_list)
            print(f'At step {global_step}, Avg. Validation AUC is {auc_val_avg:.4f}')
            
            if args.use_tb:
                writer.add_scalar("val/val_avg_auc", auc_val_avg, global_step)

        if (global_step+1) % args.save_frequency == 0 and global_step>=args.learning_starts:
            directory = os.path.join('saved', run_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({"policy_state_dict": policy.state_dict(), 
                "qf1_state_dict": qf1.state_dict(),
                "qf2_state_dict": qf2.state_dict(),
                "qf1_target_state_dict": qf1_target.state_dict(),
                "qf2_target_state_dict": qf2_target.state_dict()
                }, os.path.join(directory, f'{global_step}.ckpt'))
            
        if (global_step + 1) % 50 == 0:
            time_relative = str(timedelta(seconds=time.time() - start_time)).split('.')[0]
            auc_avg = sum(auc_buffer)/len(auc_buffer)
            print(f"[{time_relative} | {num_eps} episodes | {global_step} steps] Avg. AUC = {auc_avg:.3f}")
            if args.use_tb:
                writer.add_scalar("train/AUC", auc_avg, global_step)
            
        if global_step > args.learning_starts:
            for _ in range(args.num_updates):
                obs_b, act_b, obs_next_b, rew_b, done_b = buffer.sample(args.batch_size)
                
                # CRITIC Training
                with torch.no_grad():
                    _, logp_next_b = policy.get_action(obs_next_b)
                    
                    qf1_next_b = qf1_target(obs_next_b)
                    qf2_next_b = qf2_target(obs_next_b)
                    qf_next_b = torch.min(qf1_next_b, qf2_next_b)-args.alpha*logp_next_b
                    
                    # use E[Q(s',a')|a'] instead of using MC
                    b = obs_next_b.batch[obs_next_b.non_omni_mask]
                    v_next_b = scatter_add(logp_next_b.exp()*qf_next_b, b, dim_size=obs_next_b.batch_size)
                    q_target_b = rew_b.flatten() + (1-done_b.flatten()) * args.gamma * v_next_b
                
                # use Q-values only for the taken actions
                act_b += obs_b.act_offsets
                q1_b = qf1(obs_b).gather(0, act_b).flatten()
                q2_b = qf2(obs_b).gather(0, act_b).flatten()
                q_loss = mse_loss(q1_b, q_target_b) + mse_loss(q2_b, q_target_b)
                
                q_optimizer.zero_grad(); q_loss.backward(); q_optimizer.step()
                
                # ACTOR Training
                _, logp_b = policy.get_action(obs_b)
                with torch.no_grad():
                    qf1_b = qf1(obs_b)
                    qf2_b = qf2(obs_b)
                v_b = logp_b.exp()*(args.alpha*logp_b - torch.min(qf1_b, qf2_b))
                b = obs_b.batch[obs_b.non_omni_mask]
                policy_loss = scatter_add(v_b, b, dim_size=obs_b.batch_size).mean()
                
                policy_optimizer.zero_grad(); policy_loss.backward(); policy_optimizer.step()
                
                if args.use_tb and num_updates%200==0:
                    writer.add_scalar("losses/q1(s,a)", q1_b.mean().item(), global_step)
                    writer.add_scalar("losses/q2(s,a)", q1_b.mean().item(), global_step)
                    writer.add_scalar("losses/q_loss", q_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/policy_loss", -policy_loss.item(), global_step)
                    
                num_updates += 1
            
            if global_step%args.target_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

# nohup python -u sac.py --use_tb --device cuda:0 > train.out 2>&1 &