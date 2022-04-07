import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder
import data_augs as rad

LOG_FREQ = 10000

        
def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_name,
        log_std_min, log_std_max, num_layers, num_filters, encoder_feature_dim,
        separable_conv, learnable_smoothing,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_name, obs_shape, num_layers,
            num_filters, encoder_feature_dim, output_logits=True,
            separable_conv=separable_conv, learnable_smoothing=learnable_smoothing,
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.reset_weights()

    def reset_weights(self):
        self.apply(weight_init)

    def forward(self, obs, obs_shifts=None, compute_pi=True,
                compute_log_pi=True, detach_encoder=False,
                aug_finalfmap=False, aug_finalfmap_detach=False,
                unaug_finalfmap=False, sample_augs=False,
    ):
        obs = self.encoder(obs, obs_shifts, detach=detach_encoder,
                           aug_finalfmap=aug_finalfmap,
                           aug_finalfmap_detach=aug_finalfmap_detach,
                           unaug_finalfmap=unaug_finalfmap,
                           sample_augs=sample_augs,
                          )

        obs = self.trunk[1](self.trunk[0](obs))
        self.outputs['fc1'] = obs
        obs = self.trunk[3](self.trunk[2](obs))
        self.outputs['fc2'] = obs
        mu, log_std = self.trunk[4](obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_name,
        num_layers, num_filters, encoder_feature_dim, separable_conv, learnable_smoothing,
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_name, obs_shape, num_layers,
            num_filters, encoder_feature_dim, output_logits=True,
            separable_conv=separable_conv, learnable_smoothing=learnable_smoothing,
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.reset_weights()

    def reset_weights(self):
        self.apply(weight_init)

    def forward(self, obs, action, obs_shifts=None, detach_encoder=False,
                aug_finalfmap=False, aug_finalfmap_detach=False,
                unaug_finalfmap=False, sample_augs=False,
               ):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, obs_shifts, detach=detach_encoder,
                           aug_finalfmap=aug_finalfmap,
                           aug_finalfmap_detach=aug_finalfmap_detach,
                           unaug_finalfmap=unaug_finalfmap,
                           sample_augs=sample_augs,
                          )

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    #def update_target(self):
    #    utils.soft_update_params(self.encoder, self.encoder_target, 0.05)

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class RadSacAgent(object):
    """RAD with SAC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_name='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        separable_conv=False,
        learnable_smoothing=False,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
        data_augs = '',
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.latent_dim = latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_name = encoder_name
        self.data_augs = data_augs

        aug_to_func = {
                'aug_discrete' : rad.random_crop,
                'aug_even' : rad.random_crop_even,
                'aug_continuous' : rad.random_crop_continuous,
                'aug_cnn' : rad.random_crop,
                'aug_mlp' : rad.random_crop_finalfmap,
                'aug_mlp_detach' : rad.random_crop_finalfmap,
                'aug_qpred' : rad.random_crop,
                'aug_qtarget' : rad.random_crop,
                'no_aug' : rad.no_aug,
            }

        self.aug_func = aug_to_func[self.data_augs]
        self.aug_obs = self.data_augs != 'aug_qtarget'
        self.aug_next_obs = self.data_augs != 'aug_qpred'
        self.aug_finalfmap = self.data_augs == 'aug_mlp'
        self.aug_finalfmap_detach = self.data_augs == 'aug_mlp_detach'
        self.unaug_finalfmap = self.data_augs == 'aug_cnn'

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_name,
            actor_log_std_min, actor_log_std_max, num_layers, num_filters,
            encoder_feature_dim, separable_conv, learnable_smoothing,
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_name,
            num_layers, num_filters, encoder_feature_dim, separable_conv, learnable_smoothing,
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_name,
            num_layers, num_filters, encoder_feature_dim, separable_conv, learnable_smoothing,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        log_alpha_data = torch.tensor(np.log(init_temperature), dtype=torch.float32, device=device)
        self.log_alpha = torch.nn.Parameter(data=log_alpha_data, requires_grad=True)

        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, info, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, info['next_obs_shifts'],
                                                     aug_finalfmap=self.aug_finalfmap,
                                                     aug_finalfmap_detach=self.aug_finalfmap_detach,
                                                     unaug_finalfmap=self.unaug_finalfmap,
                                                     sample_augs=True,
                                                    )
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action,
                                                      info['next_obs_shifts'],
                                                      aug_finalfmap=self.aug_finalfmap,
                                                      aug_finalfmap_detach=self.aug_finalfmap_detach,
                                                      unaug_finalfmap=self.unaug_finalfmap,
                                                      sample_augs=True,
                                                     )
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, info['obs_shifts'], detach_encoder=self.detach_encoder,
            aug_finalfmap=self.aug_finalfmap, aug_finalfmap_detach=self.aug_finalfmap_detach,
            unaug_finalfmap=self.unaug_finalfmap, sample_augs=True,
        )

        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, info, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, info['obs_shifts'], detach_encoder=True,
                                            aug_finalfmap=self.aug_finalfmap,
                                            aug_finalfmap_detach=self.aug_finalfmap_detach,
                                            unaug_finalfmap=self.unaug_finalfmap,
                                            sample_augs=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, info['obs_shifts'], detach_encoder=True,
                                         aug_finalfmap=self.aug_finalfmap,
                                         aug_finalfmap_detach=self.aug_finalfmap_detach,
                                         unaug_finalfmap=self.unaug_finalfmap,
                                         sample_augs=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, info = replay_buffer.sample_rad(self.aug_func,
                                                                                 aug_obs=self.aug_obs,
                                                                                 aug_next_obs=self.aug_next_obs)
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, info, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, info, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_latest(self, model_dir):
        torch.save(
            self.actor.state_dict(), '%s/actor.pt' % model_dir
        )
        # torch.save(
            # self.critic.state_dict(), '%s/critic.pt' % model_dir
        # )

    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step), map_location=self.device)
            # torch.load('%s/actor.pt' % model_dir, map_location=self.device)
        )
        # self.critic.load_state_dict(
            # torch.load('%s/critic_%s.pt' % (model_dir, step), map_location=self.device)
        # )
 
