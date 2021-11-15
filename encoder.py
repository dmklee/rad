import torch
import torch.nn as nn
from antialiased_cnns import BlurPool

from equi_utils import affine2d

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}
OUT_DIM_108 = {4: 47}

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self,
                 obs_shape,
                 fmap_shifts,
                 feature_dim,
                 num_layers=2,
                 num_filters=32,
                 output_logits=False,
                 dropout='',
                ):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # try 2 5x5s with strides 2x2. with samep adding, it should reduce 84 to 21, so with valid, it should be even smaller than 21.
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        self.fmap_shifts = [None for _ in range(num_layers)]
        if fmap_shifts != '':
            for i, a in enumerate(fmap_shifts.split(':')):
                if a == '':
                    continue
                if ',' in a:
                    t,r = [float(z) for z in a.split(',')]
                    self.fmap_shifts[i] = (t, r)
                else:
                    a = float(a)
                    self.fmap_shifts[i] = (a, 0)
        self.fmap_dropouts = [None for _ in range(num_layers)]
        if dropout != '':
            for i,a in enumerate(dropout.split(':')):
                if a != '':
                    self.fmap_dropouts[i] = nn.Dropout2d(float(a))

        print(self.fmap_dropouts)

        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs, sample_augs=False):
        if obs.max() > 1.:
            obs = obs / 255.

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv
        if sample_augs and self.fmap_shifts[0] is not None:
            dhw = self.fmap_shifts[0][0] * (2*torch.rand(size=(obs.size(0), 2), device=obs.device)-1)
            dth = self.fmap_shifts[0][1] * (2*torch.rand(size=(obs.size(0), 1), device=obs.device)-1)
            conv = affine2d(conv, dhw, dth)
        if sample_augs and self.fmap_dropouts[0] is not None:
            conv = self.fmap_dropouts[0](conv)

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv
            if sample_augs and self.fmap_shifts[i] is not None:
                dhw = self.fmap_shifts[0][0] * (2*torch.rand(size=(obs.size(0), 2), device=obs.device)-1)
                dth = self.fmap_shifts[0][1] * (2*torch.rand(size=(obs.size(0), 1), device=obs.device)-1)
                conv = affine2d(conv, dhw, dth)
            if sample_augs and self.fmap_dropouts[i] is not None:
                conv = self.fmap_dropouts[i](conv)

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False, sample_augs=False):
        h = self.forward_conv(obs, sample_augs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc0'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln0'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

class AntiAliasedPixelEncoder(PixelEncoder):
    def __init__(self, obs_shape, fmap_shifts, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super(PixelEncoder, self).__init__()

        if fmap_shifts != '':
            print('[WARNING]: internal data aug not supported for antialiased CNN')

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=1)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.blurpool = BlurPool(num_filters, stride=2)

        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def forward_conv(self, obs, sample_augs=False):
        if obs.max() > 1.:
            obs = obs / 255.

        self.outputs['obs'] = obs

        # conv = torch.relu(self.blurpool(self.convs[0](obs)))
        conv = self.blurpool(torch.relu(self.convs[0](obs)))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, fmap_shifts, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixel_aa': AntiAliasedPixelEncoder,
                       'identity': IdentityEncoder}

def make_encoder(
    encoder_type, obs_shape, fmap_shifts, feature_dim, num_layers, num_filters, output_logits=False, dropout=0.,
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, fmap_shifts, feature_dim, num_layers, num_filters, output_logits, dropout,
    )
