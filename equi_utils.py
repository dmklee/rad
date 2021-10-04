import torch
import numpy as np
import torch.nn as nn
import torchextractor as tx

def _measure_equivariance(x, f, T_input, T_output=nn.Identity):
    '''
    x : input tensor
    f : nn module that processes x
    T_input : input transformation
    T_output : output transformation, default to identity
    '''
    A = f( T_input(x) ).view(x.size(0),-1)
    B = T_output( f(x) ).view(x.size(0),-1)
    return nn.CosineSimilarity()(A, B)

def eval_model_invariance(model, x, aug_fn, n_augs=16):
    '''evaluates the invaraince of a model at every feature map
    '''
    module_names = tx.list_module_names(model)
    tx_model = tx.Extractor(model, module_names)

    _, unaug_features = tx_model(x)

    x_aug = torch.cat([aug_fn(x) for _ in range(n_augs)], dim=0)
    _, aug_features = tx_model(x_aug)

    metric = nn.CosineSimilarity(dim=1)

    results = {}
    for name in aug_features.keys():
        unaug_fmap = unaug_features[name].repeat(n_augs)
        aug_fmap = aug_features[name]
        inv = metric(unaug_fmap, aug_fmap).mean()

        results[name] = inv

    return results

