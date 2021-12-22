import numpy as np
import torch
import torch.nn as nn
from antialiased_cnns import BlurPool

from equi_utils import affine2d


class SpatialMax2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # assumes x is shape (B, C, H, W)
        return torch.max(torch.flatten(x,2), dim=-1)[0]

class Index2d(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.theta = torch.tensor(((1,0,0),(0,1,0)), dtype=torch.float32)

    def forward(self, x):
        # input is (B,C,H,W), output is (B,C+2,H,W)
        theta = self.theta.to(x.device).repeat(x.size(0), 1,1)
        coords = torch.nn.functional.affine_grid(theta, x.size(), align_corners=True).permute(0,3,1,2)
        return torch.concat([x, coords], dim=1)

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

def parse_fmap_shifts(fmap_shifts, num_layers):
    shifts = [None for _ in range(num_layers)]
    if fmap_shifts != '':
        for layer_id, desc in enumerate(fmap_shifts.split(':')):
            if desc != '':
                if ',' in desc:
                    shifts[layer_id] = tuple([float(z) for z in desc.split(',')])
                else:
                    shifts[layer_id] = (float(desc), 0)

    return shifts

def parse_dropout(dropout, num_layers):
    layers = [None for _ in range(num_layers)]
    for layer_id, a in enumerate(dropout.split(':')):
        if a != '':
            layers[layer_id] = nn.Dropout2d(float(a))
    return layers

def create_cnn(in_channels, filters_per_conv, downsampling_per_conv, downsampling_mode):
    num_layers = len(filters_per_conv)
    strides = downsampling_per_conv if downsampling_mode == 'stride' else num_layers*[1]

    channels_list = [in_channels] + filters_per_conv
    convs = nn.ModuleList(
        [nn.Conv2d(channels_list[i], channels_list[i+1], 3, stride=strides[i]) for i in range(num_layers)]
    )
    if downsampling_mode == 'stride':
        downsamplers = nn.ModuleList([nn.Identity() for i in range(num_layers)])
    elif downsampling_mode == 'blurpool':
        tmp = []
        for n_filters, ds in zip(filters_per_conv, downsampling_per_conv):
            tmp.append(BlurPool(n_filters, stride=ds) if ds != 1 else nn.Identity())
            # tmp.append(nn.MaxPool2d(ds, ds, ceil_mode=True) if ds != 1 else nn.Identity())
        downsamplers = nn.ModuleList(tmp)

    return convs, downsamplers

def create_projector(input_shape, output_dim, projection_style,
                     prepooling_factor, projection_dim,
                     indexed_projection):
    '''projects from feature map of last conv to feature_dim'''
    if projection_style == 'mlp':
        projector = nn.Sequential(nn.Flatten(1),
                                  nn.Linear(np.product(input_shape), output_dim))
    elif projection_style == 'e2i':
        layers = []
        if prepooling_factor > 1:
            layers.append(nn.AvgPool2d(prepooling_factor, prepooling_factor))
        in_channels = input_shape[0]
        if indexed_projection:
            in_channels += 2
            layers.append(Index2d(input_shape[1]))
        layers.append(nn.Conv2d(in_channels, projection_dim, 1, stride=1))
        layers.append(SpatialMax2d())
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(projection_dim, output_dim))
        projector = nn.Sequential(*layers)
    else:
        raise TypeError('invalid value for projection_style')

    ln = nn.LayerNorm(output_dim)

    return projector, ln

def calc_cnn_output_shape(obs_shape, filters_per_conv, downsampling_per_conv,
                         k_size=3, padding=0):
    # assumes kernel size of 3
    size = obs_shape[-1]
    for ds in downsampling_per_conv:
        size = int((size + 2*padding - k_size)/ds + 1)
    return (filters_per_conv[-1], size, size)

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self,
                 obs_shape,
                 fmap_shifts,
                 feature_dim,
                 filters_per_conv,
                 downsampling_per_conv,
                 downsampling_mode='stride',
                 projection_style='mlp',
                 output_logits=False,
                 dropout='',
                 prepooling_factor=1,
                 projection_dim=2056,
                 indexed_projection=True,
                ):
        # print('filters_per_conv',filters_per_conv)
        # print('downsampling_per_conv',downsampling_per_conv)
        assert downsampling_mode in ('stride', 'blurpool')
        assert projection_style in ('mlp', 'e2i')
        super().__init__()
        # self.ds_factors = {f'conv{i+1}':downsampling_per_conv[i] for i in range(len(downsampling_per_conv))}
        self.ds_factors = {f'conv{i+1}':ds for i,ds in enumerate(np.cumprod(downsampling_per_conv))}

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim # output feature dimension
        self.num_layers = len(filters_per_conv)

        self.convs, self.downsamplers = create_cnn(in_channels=obs_shape[0],
                                                   filters_per_conv=filters_per_conv,
                                                   downsampling_per_conv=downsampling_per_conv,
                                                   downsampling_mode=downsampling_mode)

        self.fmap_shifts = parse_fmap_shifts(fmap_shifts, self.num_layers)
        self.fmap_dropouts = parse_dropout(dropout, self.num_layers)

        input_shape = calc_cnn_output_shape(obs_shape, filters_per_conv, downsampling_per_conv)
        self.projector, self.ln = create_projector(input_shape=input_shape,
                                                   output_dim=feature_dim,
                                                   projection_style=projection_style,
                                                   prepooling_factor=prepooling_factor,
                                                   projection_dim=projection_dim,
                                                   indexed_projection=indexed_projection)

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

        conv = obs
        for i in range(self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            conv = self.downsamplers[i](conv)
            self.outputs[f'conv{i+1}'] = conv
            if sample_augs:
                if self.fmap_shifts[i] is not None:
                    dhw = self.fmap_shifts[i][0] * (2*torch.rand(size=(obs.size(0), 2), device=obs.device)-1)
                    dth = self.fmap_shifts[i][1] * (2*torch.rand(size=(obs.size(0), 1), device=obs.device)-1)
                    conv = affine2d(conv, dhw, dth)
                if self.fmap_dropouts[i] is not None:
                    conv = self.fmap_dropouts[i](conv)

        return conv

    def forward(self, obs, detach=False, sample_augs=False):
        h = self.forward_conv(obs, sample_augs)

        if detach:
            h = h.detach()

        h_fc = self.projector(h)
        self.outputs['fc0'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln0'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def freeze_conv_weights(self):
        for param in self.convs.parameters():
            param.requires_grad = False

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
        # L.log_param('train_encoder/proj', self.projector, step)
        L.log_param('train_encoder/ln', self.ln, step)


_AVAILABLE_ENCODERS = ('pixel',
                       'pixel_narrow',
                       'pixel_narrow_e2i',
                       'pixel_narrow_e2i_ind',
                       'pixel_aa',
                       # 'pixel_narrow_aa',
                       # 'pixel_narrow_aa_e2i',
                       # 'pixel_narrow_aa_e2i_ind'
                      )

def make_encoder(encoder_type,
                 obs_shape,
                 fmap_shifts,
                 feature_dim,
                 num_layers,
                 num_filters,
                 output_logits=False,
                 dropout='',
):
    assert encoder_type in _AVAILABLE_ENCODERS

    downsampling_mode = 'blurpool' if encoder_type.find('aa') > -1 else 'stride'
    projection_style = 'e2i' if encoder_type.find('e2i') > -1 else 'mlp'
    if encoder_type.find('narrow') > -1:
        downsampling_per_conv = 3*[2] + (num_layers-3)*[1]
        filters_per_conv = [num_filters * ds for ds in np.cumprod(downsampling_per_conv)]
    else:
        downsampling_per_conv = [2] + (num_layers-1)*[1]
        filters_per_conv = num_layers * [num_filters]

    indexed_projection = encoder_type.find('ind') > -1

    return PixelEncoder(obs_shape,
                        fmap_shifts,
                        feature_dim,
                        filters_per_conv,
                        downsampling_per_conv,
                        downsampling_mode=downsampling_mode,
                        projection_style=projection_style,
                        output_logits=output_logits,
                        dropout=dropout,
                        prepooling_factor=1,
                        projection_dim=2056,
                        indexed_projection=indexed_projection)

if __name__ == "__main__":
    import time
    # pixel
    device = torch.device('cuda')
    obs_shape = (9,84,84)
    B = 512
    obs = torch.rand((B, *obs_shape)).to(device)

    num_trials = 50
    data = {}
    feature_dim = 50
    for encoder_type in _AVAILABLE_ENCODERS:
        encoder = make_encoder(encoder_type, obs_shape, '', feature_dim, 4, 32).to(device)
        optim = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        fp_time = 0
        bp_time = 0
        for _ in range(num_trials):
            t = time.time()
            out = encoder(obs)
            fp_time += time.time()-t

            t = time.time()
            optim.zero_grad()
            loss = out.sum()
            loss.backward()
            optim.step()
            bp_time += time.time()-t

        data[encoder_type] = {'num params' : sum(p.numel() for p in encoder.parameters())/1000000,
                              'foward pass (ms)' : 1000*fp_time/num_trials,
                              'backward pass (ms)' : 1000*bp_time/num_trials,}

    for k, v in data.items():
        print(f' ===== {k} ===== ')
        print(v)


