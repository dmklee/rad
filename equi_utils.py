import os
import numpy as np
import torch
import json
import numpy as np
import torch.nn as nn
import torchextractor as tx
import dmc2gym

import utils
from train import make_agent

def cosine_distance(x, y):
    cosine_sim = nn.CosineSimilarity()(x.reshape(x.size(0),-1),
                                       y.reshape(y.size(0),-1))
    return 1 - cosine_sim

def shift2d(x, dhw):
    '''
    dh -> shift up (+); shift down (-)
    dw -> shift left (+); shift right (-)
    '''
    N = x.shape[0]
    size = x.shape[-1]
    arange = torch.linspace(-1, 1, size, device=x.device, dtype=torch.float32)
    arange = arange.unsqueeze(0).repeat(size, 1).unsqueeze(2)
    grid = torch.cat([arange, arange.transpose(1,0)], dim=2)
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)
    grid[...,0] += 2.0 * dhw[:,0].view(N, 1, 1)/size
    grid[...,1] += 2.0 * dhw[:,1].view(N, 1, 1)/size

    unshifted = nn.functional.grid_sample(x, grid,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=True)
    return unshifted

def functional_shift(dhw):
    def thunk(x):
        return shift2d(x, dhw)
    return thunk

def _sample_pixel_shifts(N, min_shift, max_shift):
    dhw = min_shift + (max_shift-min_shift) * torch.rand(size=(N,2))

    return dhw

def _eval_model(model, imgs, shift_range, n_augs=8):
    '''evaluates the invaraince of a model at every feature map
    '''
    # create extractor for recording intermediate feature maps
    device = next(model.parameters()).device
    inv_module_names = extract_module_names(model, 'inv')
    equiv_module_names = extract_module_names(model, 'equiv')
    tx_model = tx.Extractor(model, inv_module_names)

    # generate inputs
    raw_x = torch.tensor(imgs, device=device).repeat(n_augs, 1,1,1)
    dhw = _sample_pixel_shifts(raw_x.size(0), *shift_range).to(device)
    shifted_x = shift2d(raw_x, dhw)

    # raw fmaps
    _, raw_fmaps = tx_model(raw_x)
    raw_fmaps = {name : f.clone().cpu() for name, f in raw_fmaps.items()}

    # shifted fmaps
    _, shifted_fmaps = tx_model(shifted_x)
    shifted_fmaps = {name : f.clone().cpu() for name, f in shifted_fmaps.items()}

    # deshifted fmaps
    dhw = dhw.cpu()
    deshifted_fmaps = {}
    ds_factor = 1
    border_size = {name : shift_range[1] for name in equiv_module_names}
    for name in equiv_module_names:
        a,b = name.split('.')
        ds_factor *= model._modules[a][int(b)].stride[0]
        border_size[name] *= ds_factor
        deshifted_fmaps[name] = shift2d(shifted_fmaps[name], -dhw/ds_factor)

    # now calculate equivariance and/or invariance
    results = {'inv' : dict(), 'equiv' : dict()}
    for name in inv_module_names:
        raw = raw_fmaps[name]
        shifted = shifted_fmaps[name]
        if name in equiv_module_names:
            output_size = int(raw.shape[-1] - 2*border_size[name])
            raw = utils.center_crop_images(raw, output_size)
            shifted = utils.center_crop_images(shifted, output_size)
            deshifted = utils.center_crop_images(deshifted_fmaps[name], output_size)
            results['equiv'][name] = cosine_distance(raw, deshifted).mean().item()

        results['inv'][name] = cosine_distance(raw, shifted).mean().item()

    return results

def load_from_results(results_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args_fname = os.path.join(results_folder, 'args.json')
    with open(args_fname, 'r') as f:
        args = utils.Bunch(json.loads(f.read()))
    utils.set_seed_everywhere(args.seed)

    pre_transform_image_size = args.pre_transform_image_size if 'crop' in args.data_augs else args.image_size
    pre_image_size = args.pre_transform_image_size # record the pre transform image size for translation

    # create environment
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat
    )

    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,
                             pre_transform_image_size,
                             pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    agent.load(os.path.join(results_folder, 'model'), 100000)

    return env, agent, args

def collect_observations(env, agent, n_obs, policy_type):
    assert policy_type in ('optimal', 'sampled', 'random')
    # collect samples
    obss = []
    done = True
    while len(obss) < n_obs:
        if done:
            obs = env.reset()
            episode_step = 0
            done = False

        with utils.eval_mode(agent):
            if policy_type == 'optimal':
                print(obs.shape, agent.image_size)
                action = agent.select_action(utils.center_crop_image(obs, agent.image_size) / 255.)
            elif policy_type == 'sampled':
                action = agent.sample_action(obs / 255.)
            else:
                action = env.sample_action()

        obs, reward, done, _ = env.step(action)
        episode_step += 1
        obss.append(obs)

    obss = np.asarray(obss).astype(np.float32)/255.
    return obss

def extract_module_names(model, mode='inv'):
    names = []
    for k, m in model._modules.items():
        if isinstance(m, nn.ModuleList):
            for i, sm in enumerate(m):
                if type(sm) == nn.Conv2d:
                    names.append(f"{k}.{i}")
                elif mode == 'inv':
                    names.append(f"{k}.{i}")
        else:
            if type(m) == nn.Conv2d:
                names.append(f"{k}")
            elif mode == 'inv':
                names.append(f"{k}")
    return names

def evaluate_models(results_folder, n_samples=128, n_augs=8, policy_type='optimal'):
    env, agent, args = load_from_results(results_folder)

    obss = collect_observations(env, agent, n_samples, policy_type)

    results = {}
    network_names = ['actor', 'critic']

    data_aug_params = [(0,2),(1,4),(4,6)]
    for network_name in network_names:
        results[network_name] = dict()
        network = agent.actor.encoder if network_name == 'actor' else agent.critic.encoder

        for params in data_aug_params:
            with torch.no_grad():
                results[network_name][params] = _eval_model(network, obss, params, n_augs)

    # metadata
    results['n_samples'] = n_samples
    results['n_augs'] = n_augs
    results['policy_type'] = policy_type

    save_path = os.path.join(results_folder, 'shiftability_data.npy')
    np.save(save_path, results, allow_pickle=True)
    return results

if __name__ == "__main__":
    parent_dir = 'results'
    folders = [os.path.join(parent_dir, p) for p in next(os.walk(parent_dir))[1]]
    for f in folders:
        if f.count('reacher') and f.count('without')<0:
            evaluate_models(f)
