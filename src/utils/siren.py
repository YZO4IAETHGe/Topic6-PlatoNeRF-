import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SIREN_NET(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, intensity=False, density_only=True):
        """ 
        """
        super(SIREN_NET, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_layers = nn.ModuleList([])
        for ind in range(D):
            is_first = ind == 0
            layer_dim_in = input_ch if is_first else W
            w = 30 if is_first else 1
            
            if ind in self.skips and not is_first:
                layer_dim_in += input_ch

            self.pts_layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = W,
                is_first = is_first,
                w0 = w
            ))
        
    
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            if density_only:
                self.rgb_linear = None
            elif intensity:
                self.rgb_linear = nn.Linear(W//2, 1)
            else:
                self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        input_pts_norm = input_pts.clone()
        grid = 10
        input_pts_norm[..., 0] = 2.0 * (input_pts[..., 0] - (-grid)) / (grid - (-grid)) - 1.0
        input_pts_norm[..., 1] = 2.0 * (input_pts[..., 1] - (-grid)) / (grid - (-grid)) - 1.0
        input_pts_norm[..., 2] = 2.0 * (input_pts[..., 2] - (-grid)) / (grid - (-grid)) - 1.0
        input_pts = input_pts_norm
       
        h = input_pts
        for i, l in enumerate(self.pts_layers):
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
            h = self.pts_layers[i](h)
            
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = torch.rand(alpha.shape)
            if self.rgb_linear is not None:
                rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_layers[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_layers[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


def exists(val):
  return val is not None
    
class Sine(nn.Module):
  def __init__(self, w0 = 1.):
    super().__init__()
    self.w0 = w0
  def forward(self, x):
    return torch.sin(self.w0 * x)

# siren layer
class Siren(nn.Module):
  def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
    super().__init__()
    self.dim_in = dim_in
    self.is_first = is_first

    weight = torch.zeros(dim_out, dim_in)
    bias = torch.zeros(dim_out) if use_bias else None
    self.init_(weight, bias, c = c, w0 = w0)

    self.weight = nn.Parameter(weight)
    self.bias = nn.Parameter(bias) if use_bias else None
    self.activation = Sine(w0) if activation is None else activation

  def init_(self, weight, bias, c, w0):
    dim = self.dim_in

    w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
    weight.uniform_(-w_std, w_std)

    if exists(bias):
        bias.uniform_(-w_std, w_std)

  def forward(self, x):
    out =  F.linear(x, self.weight, self.bias)
    out = self.activation(out)
    return out
