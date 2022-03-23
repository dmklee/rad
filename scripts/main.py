import os
import argparse
import json
import time
import numpy as np
import random
import torch
import dmc2gym

import utils
from logger import Logger
from curl_sac import RadSacAgent

from train import evaluate, make_agent

ACTION_REPEAT_LOOKUP = {('finger', 'spin') : 2,
                        ('walker', 'walk') : 2,
                        ('cartpole', 'swingup') : 8,
                        ('reacher', 'easy') : 4,
                        ('cheetah', 'run') : 4,
                        ('ball_in_cup', 'catch') : 4,
                       }

def add_defaults(args):
    args.domain_name, args.task_name = args.env.split('-')
    args.pre_transform_image_size = 100
    args.image_size = 84
    args.action_repeat = ACTION_REPEAT_LOOKUP[tuple(args.env.split('-'))]
    args.frame_stack = 3
    args.replay_buffer_capacity = 100000
    args.init_steps = 1000
    args.num_updates_per_env_step = 1
    args.batch_size = 512
    args.hidden_dim = 1024
    args.eval_freq = 5000
    args.num_eval_episodes = 100
    args.critic_lr = 2e-4 if args.env == 'cheetah-run' else 1e-3
    args.critic_beta = 0.9
    args.critic_tau = 0.01
    args.critic_target_update_freq = 2
    args.actor_lr = 2e-4 if args.env == 'cheetah-run' else 1e-3
    args.actor_beta = 0.9
    args.actor_log_std_min = -10
    args.actor_log_std_max = 2
    args.actor_update_freq = 2
    args.encoder_feature_dim = 50
    args.encoder_lr = 2e-4 if args.env == 'cheetah-run' else 1e-3
    args.encoder_tau = 0.05
    args.num_layers = 4
    args.num_filters = 32
    args.latent_dim = 50
    args.discount = 0.99
    args.init_temperature = 0.1
    args.alpha_lr = 1e-4
    args.alpha_beta = 0.5
    args.save_tb = False
    args.save_buffer = False
    args.save_video = False
    args.save_model = False
    args.detach_encoder = False
    args.log_interval = 100
    return args


def create_env_buffer_agent(args, device):
    pre_transform_image_size = args.pre_transform_image_size if args.data_augs!='no_aug' else args.image_size

    env = dmc2gym.make(domain_name=args.domain_name,
                       task_name=args.task_name,
                       seed=args.seed,
                       visualize_reward=False,
                       from_pixels=(args.encoder_name.find('pixel')>=0),
                       height=pre_transform_image_size,
                       width=pre_transform_image_size,
                       frame_skip=args.action_repeat)
    action_shape = env.action_space.shape

    if args.encoder_name.find('pixel')>=0:
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,
                             pre_transform_image_size,
                             pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(obs_shape=pre_aug_obs_shape,
                                       action_shape=action_shape,
                                       capacity=args.replay_buffer_capacity,
                                       batch_size=args.batch_size,
                                       device=device,
                                       image_size=args.image_size,
                                       pre_image_size=args.pre_transform_image_size)

    agent = make_agent(obs_shape=obs_shape,
                       action_shape=action_shape,
                       args=args,
                       device=device)

    return env, replay_buffer, agent


def generate_mini_checkpoint(step, args):
    torch.save({
        'step' : step,
    }, os.path.join(args.work_dir, 'mini_checkpoint.pt'))

def generate_checkpoint(env, replay_buffer, agent,
                        episode, step, args):
    generate_mini_checkpoint(step, args)
    torch.save({
        'step' : step,
        'episode' : episode,

        # # replay buffer
        'replay_buffer_idx' : replay_buffer.idx,
        'replay_buffer_full' : replay_buffer.full,
        'replay_buffer_obses' : torch.tensor(replay_buffer.obses),
        'replay_buffer_next_obses' : torch.tensor(replay_buffer.next_obses),
        'replay_buffer_actions' : torch.tensor(replay_buffer.actions),
        'replay_buffer_rewards' : torch.tensor(replay_buffer.rewards),
        'replay_buffer_not_dones' : torch.tensor(replay_buffer.not_dones),

        # models and optimizers
        'critic_state_dict' : agent.critic.state_dict(),
        'critic_optimizer_state_dict' : agent.critic_optimizer.state_dict(),
        'critic_target_state_dict' : agent.critic_target.state_dict(),
        'actor_state_dict' : agent.actor.state_dict(),
        'actor_optimizer_state_dict' : agent.actor_optimizer.state_dict(),
        'log_alpha' : agent.log_alpha.data,
        'log_alpha_optimizer_state_dict' : agent.log_alpha_optimizer.state_dict(),

        # random states
        'numpy_rng_state' : np.random.get_state(),
        'torch_rng_state' : torch.get_rng_state(),
        'torch_cuda_rng_state' : torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.get_rng_state(),
        'random_rng_state' : random.getstate(),
    }, os.path.join(args.work_dir, 'checkpoint.pt'))


def load_checkpoint(exp_path, mini=False):
    if mini:
        checkpoint_path = os.path.join(exp_path, 'mini_checkpoint.pt')
    else:
        checkpoint_path = os.path.join(exp_path, 'checkpoint.pt')

    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_start_time = time.time()
    # create directory string
    exp_name = f"{args.domain_name}_{args.task_name}" \
               f"-s{args.seed}-{int(args.num_train_steps/1000)}k" \
               f"-{args.data_augs.replace('-','_') if args.data_augs != '' else 'no_aug'}" \
               f"-{args.encoder_name}"
    if args.separable_conv:
        exp_name += '-separable'
    args.work_dir = os.path.join(args.results_dir, exp_name)

    utils.set_seed_everywhere(args.seed)
    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))

    env, replay_buffer, agent = create_env_buffer_agent(args, device)
    env.seed(args.seed)
    if args.encoder_name.find('pixel')>=0:
        env = utils.FrameStack(env, k=args.frame_stack)

    ################################################################
    ## check for consistency
    ################################################################
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    ################################################################


    L = Logger(args.work_dir, use_tb=args.save_tb, remove_old=False)
    episode, episode_reward, done = 0, 0, True
    step = 0

    if os.path.exists(os.path.join(args.work_dir, 'mini_checkpoint.pt')):
        steps_completed = load_checkpoint(args.work_dir, mini=True)['step']
        if steps_completed >= args.num_train_steps:
            # no need to keep training, but remove checkpoint if it exists
            if os.path.exists(os.path.join(args.work_dir, 'checkpoint.pt')):
                os.remove(os.path.join(args.work_dir, 'checkpoint.pt'))

            exit()

        checkpoint = load_checkpoint(args.work_dir)
        step = checkpoint['step']
        episode = checkpoint['episode']-1

        # populate buffer
        replay_buffer.idx = checkpoint['replay_buffer_idx']
        replay_buffer.full = checkpoint['replay_buffer_full']
        replay_buffer.obses = checkpoint['replay_buffer_obses'].numpy()
        replay_buffer.next_obses = checkpoint['replay_buffer_next_obses'].numpy()
        replay_buffer.actions = checkpoint['replay_buffer_actions'].numpy()
        replay_buffer.rewards = checkpoint['replay_buffer_rewards'].numpy()
        replay_buffer.not_dones = checkpoint['replay_buffer_not_dones'].numpy()

        # load models
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        with torch.no_grad():
            agent.log_alpha.copy_(checkpoint['log_alpha'])
        agent.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer_state_dict'])

        # reset rng states
        np.random.set_state(checkpoint['numpy_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        agent.train()
        agent.critic_target.train()

    # begin/continue training
    start_time = time.time()
    for step in range(step, args.num_train_steps+1):
        # evaluate agent periodically
        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, None, args.num_eval_episodes, L, step, args)
            if args.save_model:
                agent.save(model_dir, step)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

            hours_elapsed = (time.time() - run_start_time)/3600
            if hours_elapsed > args.time_limit:
                # checkpoint
                print('stopping due to time limit')
                generate_checkpoint(env, replay_buffer, agent, episode, step, args)
                exit()


        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs / 255.)

        # run training update
        if step >= args.init_steps:
            agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1

    L.dump(step)
    agent.save(model_dir, args.num_train_steps)

    # in case it finished in one go, store checkpoint so repeats will know
    generate_mini_checkpoint(step, args)

    checkpoint_path = os.path.join(args.work_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        # delete big checkpoint to save memory
        os.remove(checkpoint_path)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='finger-spin')
    parser.add_argument('--data-augs', type=str, default='no_aug',
                       choices=['aug_discrete', 'aug_continuous', 'aug_even',
                                'aug_cnn', 'aug_mlp', 'aug_qpred', 'aug_qtarget',
                                'no_aug'])
    parser.add_argument('--encoder-name', type=str, default='pixel')
    parser.add_argument('--separable-conv', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-train-steps', type=int, default=100000)
    parser.add_argument('--time-limit', type=float, default=7,
                       help='max time allowed to train (in hours)')
    parser.add_argument('--results-dir', type=str, default='./test_results',
                        help='folder where results are saved')
    args = parser.parse_args()

    args = add_defaults(args)

    run(args)

