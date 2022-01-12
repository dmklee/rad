import argparse
import time
import datetime

from train import main, arg_parser

ENVS = [('finger','spin'), ('cartpole','swingup'),
        ('reacher','easy'),('cheetah','run'),
        ('walker','walk'),('cup','catch')]
ACTION_REPEAT_LOOKUP = {('finger', 'spin') : 2,
                        ('walker', 'walk') : 2,
                        ('cartpole', 'swingup') : 8,
                        ('reacher', 'easy') : 4,
                        ('cheetah', 'run') : 4,
                        ('cup', 'catch') : 4,
                       }
IMAGE_SIZE_LOOKUP = {'crop' : 84, 'translate' : 108, '' :  84}

def run_experiment(**kwargs):
    argv = []
    for k, v in kwargs.items():
        if type(v) == bool:
            # handle store true args
            if v:
                argv.append(f"--{k}")
        else:
            argv.extend([f"--{k}", f"{v}"])

    args = arg_parser().parse_args(argv)

    t = time.time()
    print(f"Running {args.agent} on {'-'.join([args.domain_name, args.task_name])} (seed={args.seed}) ")
    main(args)
    dt = time.time() - t
    print(f"Completed in {datetime.timedelta(seconds=time.time()-t)}")

def run(num_train_steps, agent, encoder_type, encoder_num_layers,
        encoder_fmap_shifts, dropout, encoder_final_fmap_dropout, encoder_final_fmap_blur,
        encoder_final_fmap_actfn, encoder_final_fmap_reg_gamma, encoder_final_fmap_reg_px,
        seeds, work_dir, envs, num_updates_per_env_step, encoder_train_steps):
    for seed in seeds:
        for env in envs:
            domain, task = env
            if agent == 'rad_sac':
                data_aug = 'crop'
            else:
                data_aug = ''
            image_size = IMAGE_SIZE_LOOKUP[data_aug]
            action_repeat = ACTION_REPEAT_LOOKUP[env]

            if encoder_train_steps is None:
                encoder_train_steps = num_train_steps

            run_experiment(domain_name=domain,
                           task_name=task,
                           pre_transform_image_size=100,
                           image_size=image_size,
                           action_repeat=action_repeat,
                           frame_stack=3,
                           replay_buffer_capacity=100000,
                           agent=agent,
                           init_steps=1000,
                           num_train_steps=num_train_steps,
                           num_updates_per_env_step=num_updates_per_env_step,
                           batch_size=512,
                           hidden_dim=1024,
                           eval_freq=10000,
                           num_eval_episodes=100,
                           critic_lr=2e-4 if env==('cheetah','run') else 1e-3,
                           critic_beta=0.9,
                           critic_tau=0.01,
                           critic_target_update_freq=2,
                           actor_lr=2e-4 if env==('cheetah','run') else 1e-3,
                           actor_beta=0.9,
                           actor_log_std_min=-10,
                           actor_log_std_max=2,
                           actor_update_freq=2,
                           encoder_type=encoder_type,
                           encoder_fmap_shifts=encoder_fmap_shifts,
                           encoder_feature_dim=50,
                           encoder_lr=2e-4 if env==('cheetah','run') else 1e-3,
                           encoder_tau=0.05,
                           encoder_dropout=dropout,
                           encoder_final_fmap_dropout=encoder_final_fmap_dropout,
                           encoder_final_fmap_blur=encoder_final_fmap_blur,
                           encoder_final_fmap_actfn=encoder_final_fmap_actfn,
                           encoder_final_fmap_reg_gamma=encoder_final_fmap_reg_gamma,
                           encoder_final_fmap_reg_px=encoder_final_fmap_reg_px,
                           num_layers=encoder_num_layers,
                           num_filters=32,
                           latent_dim=50,
                           discount=0.99,
                           init_temperature=0.1,
                           alpha_lr=1e-4,
                           alpha_beta=0.5,
                           seed=seed,
                           work_dir=work_dir,
                           save_tb=False,
                           save_buffer=False,
                           save_video=False,
                           save_model=False,
                           detach_encoder=False,
                           data_augs=data_aug,
                           encoder_train_steps=encoder_train_steps,
                          )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='rad_sac') # pixel_sac
    parser.add_argument('--encoder-type', type=str, default='pixel') # pixel_aa
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--fmap-shifts', type=str, default='')
    parser.add_argument('--dropout', type=str, default='')
    parser.add_argument('--encoder-final-fmap-dropout', type=float, default=0.)
    parser.add_argument('--encoder-final-fmap-blur', type=float, default=0.)
    parser.add_argument('--encoder-final-fmap-actfn', type=str, default='relu')
    parser.add_argument('--encoder-final-fmap-reg-gamma', type=float, default=0.)
    parser.add_argument('--encoder-final-fmap-reg-px', type=int, default=0)
    parser.add_argument('--encoder-train-steps', type=str, default=None)
    parser.add_argument('--num-train-steps', type=int, default=100000)
    parser.add_argument('--num-updates-per-env-step', type=int, default=1)
    parser.add_argument('--envs', type=str, nargs='+', default=[])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--work-dir', type=str, default='./results')
    args = parser.parse_args()

    if len(args.envs):
        envs = [tuple(e.split('-')) for e in args.envs]
    else:
        envs = ENVS

    run(args.num_train_steps, args.agent, args.encoder_type, args.num_layers,
        args.fmap_shifts, args.dropout, args.encoder_final_fmap_dropout, args.encoder_final_fmap_blur,
        args.encoder_final_fmap_actfn, args.encoder_final_fmap_reg_gamma, args.encoder_final_fmap_reg_px,
        args.seeds, args.work_dir, envs, args.num_updates_per_env_step, args.encoder_train_steps)
