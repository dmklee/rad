import os
import numpy as np
import torch
import json
import numpy as np
import torch.nn as nn
import dmc2gym

import utils

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

def affine2d(x, dhw, dth):
    '''
    dh -> shift up (+); shift down (-)
    dw -> shift left (+); shift right (-)
    dth -> rotate
    '''
    N = x.shape[0]
    size = x.shape[-1]
    affine_matrix = torch.zeros((N, 2, 3), device=x.device, dtype=torch.float32)
    affine_matrix[:,0,0] = torch.cos(dth).squeeze()
    affine_matrix[:,0,1] = -torch.sin(dth).squeeze()
    affine_matrix[:,0,2] = 2 * dhw[:,0] / size
    affine_matrix[:,1,0] = torch.sin(dth).squeeze()
    affine_matrix[:,1,1] = torch.cos(dth).squeeze()
    affine_matrix[:,1,2] = 2 * dhw[:,1] / size

    grid = torch.nn.functional.affine_grid(affine_matrix, x.size(),
                                           align_corners=True)

    unshifted = nn.functional.grid_sample(x,
                                          grid,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=True)
    return unshifted

def functional_shift(dhw):
    def thunk(x):
        return shift2d(x, dhw)
    return thunk

def sample_pixel_shifts(N, min_shift, max_shift):
    dhw = min_shift + (max_shift-min_shift) * torch.rand(size=(N,2))
    return dhw

def cosine_distance(x, y):
    cosine_sim = nn.CosineSimilarity()(x.reshape(x.size(0),-1),
                                       y.reshape(y.size(0),-1))
    return 1 - cosine_sim

def _eval_model(model, imgs, shift_range, n_augs=8):
    '''evaluates the invariance of a model at every feature map
    '''
    # create extractor for recording intermediate feature maps
    device = next(model.parameters()).device

    # generate inputs
    raw_x = torch.tensor(imgs, device=device).repeat(n_augs, 1,1,1)
    dhw = sample_pixel_shifts(raw_x.size(0), *shift_range).to(device)
    shifted_x = shift2d(raw_x, dhw)

    obs_size = model.encoder.obs_shape[-1]
    raw_x = utils.center_crop_images(raw_x, obs_size)
    shifted_x = utils.center_crop_images(shifted_x, obs_size)

    # raw fmaps
    model(raw_x)
    raw_fmaps = {name : f.clone().cpu() for name, f in model.encoder.outputs.items() if name not in ('obs', 'ln0')}
    raw_fmaps.update({name : f.clone().cpu() for name, f in model.outputs.items() if name!='std'})

    # shifted fmaps
    model(shifted_x)
    shifted_fmaps = {name : f.clone().cpu() for name, f in model.encoder.outputs.items() if name not in ('obs', 'ln0')}
    shifted_fmaps.update({name : f.clone().cpu() for name, f in model.outputs.items() if name!='std'})

    # deshifted fmaps
    dhw = dhw.cpu()
    deshifted_fmaps = {}
    ds_factor = 2
    border_size = ds_factor*shift_range[1]
    for name in raw_fmaps.keys():
        if 'conv' in name:
            deshifted_fmaps[name] = shift2d(shifted_fmaps[name], -dhw/ds_factor)

    # now calculate equivariance and/or invariance
    results = {'inv' : dict(), 'equiv' : dict()}
    for name in raw_fmaps.keys():
        raw = raw_fmaps[name]
        shifted = shifted_fmaps[name]
        if 'conv' in name:
            output_size = int(raw.shape[-1] - 2*border_size)
            raw = utils.center_crop_images(raw, output_size)
            shifted = utils.center_crop_images(shifted, output_size)
            deshifted = utils.center_crop_images(deshifted_fmaps[name], output_size)
            results['equiv'][name] = cosine_distance(raw, deshifted).mean().item()

        results['inv'][name] = cosine_distance(raw, shifted).mean().item()

    return results

def load_from_results(results_folder, is_trained=True):
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
        from_pixels=(args.encoder_type.find('pixel') > -1),
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat
    )

    if args.encoder_type.find('pixel') > -1:
        env = utils.FrameStack(env, k=args.frame_stack)

    action_shape = env.action_space.shape

    if args.encoder_type.find('pixel') > -1:
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,
                             pre_transform_image_size,
                             pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    from train import make_agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    agent.load(os.path.join(results_folder, 'model'), 100000)
    if not is_trained:
        agent.actor.reset_weights()
        agent.critic.reset_weights()

    return env, agent, args

def collect_observations(env, agent, n_obs, data_distribution):
    assert data_distribution in ('on_policy', 'off_policy')
    # collect samples
    obss = []
    done = True
    while len(obss) < n_obs:
        if done:
            obs = env.reset()
            episode_step = 0
            done = False

        with utils.eval_mode(agent):
            if data_distribution == 'on_policy':
                action = agent.select_action(utils.center_crop_image(obs, agent.image_size) / 255.)
            elif data_distribution == 'off_policy':
                action = env.action_space.sample()
            else:
                raise TypeError

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

def fraction_lowpass(kernel, padding=10):
    exp_kernel = torch.nn.functional.pad(kernel, (0,padding,0,padding))
    size = exp_kernel.shape[-1]
    exp_kernel = exp_kernel.view(-1, size, size)

    fft_kernel = torch.fft.fft2(exp_kernel)
    mask = torch.outer(torch.fft.fftfreq(size).abs() < 0.25,
                       torch.fft.fftfreq(size).abs() < 0.25).view(1,1,size,size)
    mask = mask.to(kernel.device)
    return (fft_kernel*mask).abs().sum(dim=(-1,-2))/ fft_kernel.abs().sum(dim=(-1,-2))

def evaluate_filters(agent):
    network = agent.actor
    filter_metrics = {"l2_norm":{}, "sum":{}, "lowpass":{}}
    for i, conv in enumerate(network.encoder.convs):
        filter_metrics["l2_norm"][f"conv{i+1}"] = torch.linalg.matrix_norm(conv.weight).mean().item()
        filter_metrics["sum"][f"conv{i+1}"] = conv.weight.sum(dim=(-1,-2)).mean().item()
        filter_metrics["lowpass"][f"conv{i+1}"] = fraction_lowpass(conv.weight).mean().item()

    filter_metrics["l2_norm"]["fc0"] = torch.linalg.norm(network.encoder.fc.weight, dim=1).mean().item()

    for i, fc in enumerate([a for a in network.trunk if isinstance(a, nn.Linear)]):
        filter_metrics["l2_norm"][f"fc{i+1}"] = torch.linalg.norm(fc.weight, dim=1).mean().item()

    return filter_metrics


def evaluate_symmetry(env, agent, args,
                      augmentations=[(0,2),(1,4),(4,6)],
                      n_samples=128,
                      n_augs=8,
                      data_distribution='on_policy',
                     ):
    obss = collect_observations(env, agent, n_samples, data_distribution)

    results = {}

    network = agent.actor

    for aug in augmentations:
        with torch.no_grad():
            results[aug] = _eval_model(network, obss, aug, n_augs)

    # metadata
    # results['n_samples'] = n_samples
    # results['n_augs'] = n_augs
    # results['policy_type'] = policy_type

    # save_path = os.path.join(results_folder, 'shiftability_data.npy')
    # np.save(save_path, results, allow_pickle=True)
    return results

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = 0*torch.randn((1,1,16,16))
    x[...,4:6,4:6] = 10
    x[...,12:14,12:14] = 10
    dhw = 4*torch.rand(size=(1,2))-2
    dhw[0,0] = 3
    dhw[0,1] = 0
    dth = torch.zeros((1,1))
    dth[0,0] = 0.5
    x_shift = shift2d(x, dhw)
    x_affine = affine2d(x, dhw, dth)

    f,ax = plt.subplots(1,3, figsize=(9,3))
    ax[0].imshow(x.squeeze())
    ax[1].imshow(x_shift.squeeze())
    ax[2].imshow(x_affine.squeeze())
    [a.axis('off') for a in ax]
    plt.show()
