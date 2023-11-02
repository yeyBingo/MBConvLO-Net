import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from kernels.kernel_points import load_kernels
from torch.nn.init import kaiming_uniform_
import math
import random
import time
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def mul_q_point(q_a, q_b, batch_size, mode):
    if mode:
        q_a = torch.reshape(q_a, [batch_size, 1, 4])
    else:
        q_b = torch.reshape(q_b, [batch_size, 1, 4])

    q_result_0 = torch.mul(q_a[:, :, 0], q_b[:, :, 0]) - torch.mul(q_a[:, :, 1], q_b[:, :, 1]) - torch.mul(
        q_a[:, :, 2], q_b[:, :, 2]) - torch.mul(q_a[:, :, 3], q_b[:, :, 3])
    q_result_0 = torch.reshape(q_result_0, [batch_size, -1, 1])

    q_result_1 = torch.mul(q_a[:, :, 0], q_b[:, :, 1]) + torch.mul(q_a[:, :, 1], q_b[:, :, 0]) + torch.mul(
        q_a[:, :, 2], q_b[:, :, 3]) - torch.mul(q_a[:, :, 3], q_b[:, :, 2])
    q_result_1 = torch.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = torch.mul(q_a[:, :, 0], q_b[:, :, 2]) - torch.mul(q_a[:, :, 1], q_b[:, :, 3]) + torch.mul(
        q_a[:, :, 2], q_b[:, :, 0]) + torch.mul(q_a[:, :, 3], q_b[:, :, 1])
    q_result_2 = torch.reshape(q_result_2, [batch_size, -1, 1])

    q_result_3 = torch.mul(q_a[:, :, 0], q_b[:, :, 3]) + torch.mul(q_a[:, :, 1], q_b[:, :, 2]) - torch.mul(
        q_a[:, :, 2], q_b[:, :, 1]) + torch.mul(q_a[:, :, 3], q_b[:, :, 0])
    q_result_3 = torch.reshape(q_result_3, [batch_size, -1, 1])

    q_result = torch.cat([q_result_0, q_result_1, q_result_2, q_result_3], dim=-1)

    return q_result  ##  B N 4


def inv_q(q):
    q = torch.squeeze(q, dim=1)

    q_2 = torch.sum(q * q, dim=-1, keepdim=True) + 1e-10
    q_ = torch.cat((q[:, 0].unsqueeze(1), -q[:, 1:4]), dim=-1)
    q_inv = q_ / q_2

    return q_inv


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]/[B, N, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].contiguous().view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = -square_distance(new_xyz, xyz)
    group_dist = sqrdists.topk(nsample).values
    group_idx = sqrdists.topk(nsample).indices

    group_idx[group_dist < -(radius ** 2)] = N
    group_first = group_idx[:, :, 0].contiguous().view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists = square_distance(new_xyz, xyz)
    # group_idx[sqrdists > radius ** 2] = N
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # group_first = group_idx[:, :, 0].contiguous().view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    # idx = query_ball_point(radius, nsample, xyz, new_xyz)
    _, idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    # new_xyz1 = new_xyz.view(B, S, 1, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.contiguous().view(B, S, 1, C)
    # grouped_xyz_norm = grouped_xyz - new_xyz1

    if points is not None:
        grouped_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]

        # new_points = torch.cat([grouped_xyz_norm, grouped_xyz], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        fps_points = new_xyz

    if returnfps:
        # return new_xyz, new_points, grouped_xyz, fps_points
        return new_xyz, new_points, grouped_xyz, idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = -square_distance(new_xyz, xyz)
    group_dist = sqrdists.topk(nsample).values
    group_idx = sqrdists.topk(nsample).indices
    return (0 - group_dist), group_idx


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            if i == 0:
                continue
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(1, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # xyz = xyz.permute(0, 2, 1)
        # if points is not None:
        #     points = points.permute(0, 2, 1)

        # if self.group_all:
        #     new_xyz, new_points = sample_and_group_all(xyz, points)
        # else:
        #     new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        B, N, C = xyz.shape

        xyz = xyz.contiguous()
        sample_idx = furthest_point_sample(xyz, self.npoint)
        sample_idx = sample_idx.long()
        new_xyz = index_points(xyz, sample_idx)

        # sample_idx = torch.linspace(0, N - 1, steps=self.npoint, dtype=torch.long).unsqueeze(0)
        # sample_idx = torch.repeat_interleave(sample_idx, repeats=B, dim=0)
        # new_xyz = index_points(xyz, sample_idx)
        # idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        _, idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.contiguous().view(B, self.npoint, 1, C)
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]

        # new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = conv(new_points)
            new_points = F.relu(bn(new_points))
            # new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        # new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, N, D]
            points2: input points data, [B, S, D]
        Return:
            new_points: upsampled points data, [B, N, D']
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        # points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)
        return new_points


class SharedMLPBlock(nn.Module):

    def __init__(self, in_channel, out_channel, momentum=0.1, group_all=False, hs_swish=True, bias=True):
        super(SharedMLPBlock, self).__init__()

        self.hs_swish = hs_swish
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 1, bias=bias)
        # self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 1, bias=bias)
        self.mlp_bns = nn.BatchNorm2d(self.out_channel, momentum=momentum)
        if self.hs_swish:
            self.swish = Swish()
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # xyz = xyz.permute(0, 2, 1)
        # if points is not None:
        #     points = points.permute(0, 2, 1)

        # if self.group_all:
        #     new_xyz, new_points = sample_and_group_all(xyz, points)
        # else:
        #     new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        # new_points = points.unsqueeze(dim=1)
        if len(points.shape) == 4:
            new_points = points.permute(0, 3, 1, 2).contiguous()  # [B, C, npoint, nsample]
        else:
            new_points = points.permute(0, 2, 1).contiguous()  # [B, C, npoint]
            new_points = new_points.unsqueeze(-1)

        new_points = self.conv1(new_points)
        # new_points = self.conv2(new_points)
        new_points = self.mlp_bns(new_points)
        if self.hs_swish:
            new_points = self.swish(new_points)

        if len(points.shape) == 4:
            new_points = new_points.permute(0, 2, 3, 1).contiguous()
        else:
            new_points = new_points.squeeze(-1)
            new_points = new_points.permute(0, 2, 1).contiguous()
        new_xyz = xyz
        return new_xyz, new_points


class SENet(nn.Module):
    def __init__(self, in_channel, out_channel, se_ratio=1 / 4, res=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.se_ratio = se_ratio

        mid_channel = max(1, int(self.in_channel * self.se_ratio))
        self.se_reduce = nn.Linear(self.in_channel, mid_channel)
        self.se_expand = nn.Linear(mid_channel, self.out_channel)
        self.res = res
        self.swish = Swish()

    def forward(self, xyz, points):
        # points = torch.cat([xyz, points], dim=-1)
        points = points.permute(0, 2, 1)
        x_squeezed = F.adaptive_avg_pool1d(points, 1).squeeze(-1)
        x_squeezed = self.se_reduce(x_squeezed)
        x_squeezed = self.swish(x_squeezed)
        x_squeezed = self.se_expand(x_squeezed)
        x_squeezed = x_squeezed.unsqueeze(-1)
        points_new = torch.sigmoid(x_squeezed) * points
        if self.res:
            points_new = points_new + points
        points_new = points_new.permute(0, 2, 1)
        return xyz, points_new

class MBConv3D(nn.Module):
    def __init__(self, in_channel, out_channel, momentum=0.1, npoint=8192, nsample=32, radius=4.0, expand_ratio=1):
        super().__init__()

        self.in_channel = in_channel + 3
        self.out_channel = out_channel
        self.expand_ratio = expand_ratio
        self.mid_channel = self.in_channel * self.expand_ratio
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        # Expansion phase (Inverted Bottleneck)
        # if self.expand_ratio != 1:
        self.mlp1 = SharedMLPBlock(in_channel=self.in_channel,
                                   out_channel=self.mid_channel,
                                   momentum=momentum,
                                   group_all=False,
                                   hs_swish=True)
        self.kpconv = KPConvSimple(in_dim=self.mid_channel,
                                   out_dim=self.mid_channel,
                                   radius=self.radius,
                                   npoint=self.npoint,
                                   nsample=self.nsample,
                                   momentum=momentum)
        self.senet = SENet(in_channel=self.mid_channel, out_channel=self.mid_channel)
        # self.senet_point = SENet_point(in_channel=self.npoint, out_channel=self.npoint)

        self.mlp2 = SharedMLPBlock(in_channel=self.mid_channel,
                                   out_channel=self.out_channel,
                                   momentum=momentum,
                                   group_all=False,
                                   hs_swish=False)
        self.dropout = nn.Dropout(0.2)
        self.swish = Swish()

    def forward(self, xyz, points):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        B, N, _ = xyz.shape

        # Sampling
        sample_idx = torch.linspace(0, N - 1, steps=self.npoint, dtype=torch.long).unsqueeze(0)
        # sample_idx = np.random.choice(N - 1, self.npoint, replace=False)
        # sample_idx = torch.tensor(sample_idx, dtype=torch.long).unsqueeze(0)
        sample_idx = torch.repeat_interleave(sample_idx, repeats=B, dim=0)
        new_xyz = index_points(xyz, sample_idx)

        # idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        _, idx = knn_point(self.nsample, xyz, new_xyz)

        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        xyz_diff = grouped_xyz - new_xyz.unsqueeze(2)
        grouped_points = index_points(points, idx)
        new_points = torch.cat([xyz_diff, grouped_points], dim=-1)
        new_xyz, new_points = self.mlp1(new_xyz, new_points)
        # new_points = torch.sum(new_points, dim=-2)
        new_xyz, new_points = self.kpconv(new_xyz, new_points, grouped_xyz, idx)
        # SENet
        _, new_points = self.senet(new_xyz, new_points)

        new_xyz, new_points = self.mlp2(new_xyz, new_points)
        # if self.expand_ratio == 1:
        #     new_points = self.dropout(new_points)
        #     new_points = new_points + points
        return new_xyz, new_points

class MBConv3D_fps(nn.Module):
    def __init__(self, in_channel, out_channel, momentum=0.1, npoint=8192, nsample=32, radius=4.0, expand_ratio=1, dp_rate=0,se_res=False):
        super().__init__()
        self.in_channel = in_channel + 3
        self.out_channel = out_channel
        self.expand_ratio = expand_ratio
        self.mid_channel = self.in_channel * self.expand_ratio
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        # Expansion phase (Inverted Bottleneck)
        # if self.expand_ratio != 1:
        self.mlp1 = SharedMLPBlock(in_channel=self.in_channel,
                                   out_channel=self.mid_channel,
                                   momentum=momentum,
                                   group_all=False,
                                   hs_swish=True)
        self.kpconv = KPConvSimple(in_dim=self.mid_channel,
                                   out_dim=self.mid_channel,
                                   radius=self.radius,
                                   npoint=self.npoint,
                                   nsample=self.nsample,
                                   momentum=momentum)
        self.senet = SENet(in_channel=self.mid_channel, out_channel=self.mid_channel,res=se_res)
        # self.senet_point = SENet_point(in_channel=self.npoint, out_channel=self.npoint)

        self.mlp2 = SharedMLPBlock(in_channel=self.mid_channel,
                                   out_channel=self.out_channel,
                                   momentum=momentum,
                                   group_all=False,
                                   hs_swish=False)
        self.dp_rate = dp_rate
        if self.dp_rate != 0:
            self.dropout = nn.Dropout(dp_rate)
        self.swish = Swish()

    def forward(self, xyz, points):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        B, N, _ = xyz.shape

        # Sampling
        # sample_idx = torch.linspace(0, N - 1, steps=self.npoint, dtype=torch.long).unsqueeze(0)
        # sample_idx = torch.repeat_interleave(sample_idx, repeats=B, dim=0)
        # new_xyz = index_points(xyz, sample_idx)
        xyz = xyz.contiguous()

        sample_idx = furthest_point_sample(xyz,self.npoint)
        sample_idx = sample_idx.long()
        new_xyz = index_points(xyz, sample_idx)

        # idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        _, idx = knn_point(self.nsample, xyz, new_xyz)

        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        xyz_diff = grouped_xyz - new_xyz.unsqueeze(2)
        grouped_points = index_points(points, idx)
        new_points = torch.cat([xyz_diff, grouped_points], dim=-1)

        mlp_start = time.time()
        new_xyz, new_points = self.mlp1(new_xyz, new_points)
        # new_points = torch.sum(new_points, dim=-2)
        kp_start = time.time()
        new_xyz, new_points = self.kpconv(new_xyz, new_points, grouped_xyz, idx)
        # SENet
        _, new_points = self.senet(new_xyz, new_points)
        mlp2_start = time.time()
        new_xyz, new_points = self.mlp2(new_xyz, new_points)
        mlp_end = time.time()
        # print('mlp:',kp_start-mlp_start,' kp:',mlp2_start-kp_start,' mlp2:',mlp_end-mlp2_start)
        # if self.expand_ratio == 1:
        #     new_points = self.dropout(new_points)
        #     new_points = new_points + points
        # if self.dp_rate != 0:
        #     new_points = self.dropout(new_points)
        return new_xyz, new_points

class FlowEmbedding(nn.Module):
    def __init__(self, in_channel, nsample, nsample_q, mlp1, mlp2, radius=1, pooling='sum', corr_func='concat'):
        super(FlowEmbedding, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs1 = nn.ModuleList()
        self.mlp_convs2 = nn.ModuleList()
        self.mlp_bns1 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()
        if corr_func == 'concat':
            last_channel1 = in_channel * 2 + 3
            last_channel2 = mlp1[-1] + in_channel + 3

        for out_channel in mlp1:
            self.mlp_convs1.append(nn.Conv2d(last_channel1, out_channel, 1, bias=False))
            self.mlp_bns1.append(nn.BatchNorm2d(out_channel))
            last_channel1 = out_channel

        for out_channel in mlp2:
            self.mlp_convs2.append(nn.Conv2d(last_channel2, out_channel, 1, bias=False))
            self.mlp_bns2.append(nn.BatchNorm2d(out_channel))
            last_channel2 = out_channel

    def forward(self, pos1, pos2, feature1, feature2):

        # """
        # Input:
        #     xyz1: (batch_size, 3, npoint)
        #     xyz2: (batch_size, 3, npoint)
        #     feat1: (batch_size, channel, npoint)
        #     feat2: (batch_size, channel, npoint)
        # Output:
        #     xyz1: (batch_size, 3, npoint)
        #     feat1_new: (batch_size, mlp[-1], npoint)
        # """
        #
        # B, N, _ = pos1.shape
        # _, idx = knn_point(self.nsample, pos1, pos2)
        #
        #
        # pos2_grouped = index_points(pos2, idx)  # [B, N, S, 3]
        # pos_diff = pos2_grouped - pos1.view(B, N, 1, -1)  # [B, N, S, 3]
        #
        # feat2_grouped = index_points(feature2, idx)  # [B, N, S, C]
        #
        # if self.corr_func == 'concat':
        #     feat_diff = torch.cat([feat2_grouped, feature1.view(B, N, 1, -1).repeat(1, 1, self.nsample, 1)], dim=-1)
        #
        # feat1_new = torch.cat([pos_diff, feat_diff], dim=-1)  # [B, N, S, 2*C+3]
        # feat1_new = feat1_new.permute(0, 3, 1, 2).contiguous()  # [B, 2*C+3, N, S]
        # for i, conv in enumerate(self.mlp_convs1):
        #     bn = self.mlp_bns1[i]
        #     feat1_new = F.relu(bn(conv(feat1_new)))
        #
        # feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]
        # feat1_new = feat1_new.permute(0, 2, 1).contiguous()
        # return pos1, feat1_new

        """
        Input:
            xyz1: (batch_size, 3, npoint)
            xyz2: (batch_size, 3, npoint)
            feat1: (batch_size, channel, npoint)
            feat2: (batch_size, channel, npoint)
        Output:
            xyz1: (batch_size, 3, npoint)
            feat1_new: (batch_size, mlp[-1], npoint)
                                    64
        """
        xyz1 = pos1
        xyz2 = pos2
        xyz1_t = xyz1.permute(0, 2, 1).contiguous()
        xyz2_t = xyz2.permute(0, 2, 1).contiguous()
        # B, N, C = pos1_t.shape

        B, N, _ = pos1.shape

        _, idx2 = knn_point(self.nsample_q, xyz2, xyz1)

        xyz2_grouped = index_points(xyz2, idx2)  # [B, N, S, 3]
        xyz_diff2 = xyz2_grouped - xyz1.view(B, N, 1, -1).contiguous()
        xyz_diff2 = xyz_diff2.permute(0, 3, 1, 2).contiguous()  # [B, 3, N, S]

        feat2_grouped = index_points(feature2, idx2)  # [B, N, S, C]
        if self.corr_func == 'concat':
            feat_diff2 = torch.cat(
                [feat2_grouped, feature2.contiguous().view(B, N, 1, -1).repeat(1, 1, self.nsample_q, 1)], dim=-1)
        feat_diff2 = feat_diff2.permute(0, 3, 1, 2).contiguous()

        feat1_new = torch.cat([xyz_diff2, feat_diff2], dim=1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.mlp_convs1):
            bn = self.mlp_bns1[i]
            feat1_new = F.relu(bn(conv(feat1_new)))
        feat1_new = torch.sum(feat1_new, dim=-1, keepdim=False)  # [B, mlp[-1], npoint]
        feat1_new = feat1_new.permute(0, 2, 1).contiguous()
        #####################
        _, idx1 = knn_point(self.nsample, xyz1, xyz1)

        xyz1_grouped = index_points(xyz1, idx1)  # [B, N, S, 3]
        xyz_diff1 = xyz1_grouped - xyz1.view(B, N, 1, -1).contiguous()  # [B, 3, N, S]
        xyz_diff1 = xyz_diff1.permute(0, 3, 1, 2).contiguous()

        feat1_grouped = index_points(feat1_new, idx1)  # [B, C, N, S]

        if self.corr_func == 'concat':
            feat_diff1 = torch.cat(
                [feat1_grouped, feature1.contiguous().view(B, N, 1, -1).repeat(1, 1, self.nsample, 1)], dim=-1)
        feat_diff1 = feat_diff1.permute(0, 3, 1, 2).contiguous()

        feat1_new = torch.cat([xyz_diff1, feat_diff1], dim=1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.mlp_convs2):
            bn = self.mlp_bns2[i]
            feat1_new = F.relu(bn(conv(feat1_new)))
        feat1_new = torch.sum(feat1_new, dim=-1, keepdim=False)  # [B, mlp[-1], npoint]

        feat1_new = feat1_new.permute(0, 2, 1).contiguous()
        return pos1, feat1_new


class KPConv(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighbors, neighb_inds, x):

        ###################
        # Offset generation
        ###################

        if self.deformable:

            # Get offsets with a KPConv that only takes part of the features
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds, x) + self.offset_bias

            if self.modulated:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)

                # Get modulations
                modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K:])

            else:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)

                # No modulations
                modulations = None

            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        # s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:, :1, :]) + 1e6), 1)

        # Get neighbor points [n_points, n_neighbors, dim]
        # neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(2)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(3)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=-1)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:

            # Save distances for loss
            self.min_d2, _ = torch.min(sq_distances, dim=1)

            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            in_range = torch.any(sq_distances < self.KP_extent ** 2, dim=2).type(torch.int32)

            # New value of max neighbors
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))

            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)

            # Gather new neighbor indices [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)

            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)

            # New shadow neighbors have to point to the last shadow point
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, -2, -1)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, -2, -1)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, -2, -1)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        # x = torch.cat((x, torch.zeros_like(x[:, :1, :])), 1)
        # neighb_x = gather(x, new_neighb_inds)
        neighb_x = x

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((0, 2, 1, 3)).contiguous()
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=1)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)


class KPConvSimple(nn.Module):

    def __init__(self, in_dim, out_dim, radius, npoint, nsample, momentum=0.1):
        """
        Initialize a simple convolution block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(KPConvSimple, self).__init__()

        # get KP_extent from current radius
        current_extent = 0.45 * 1.2 / 2.5

        # Get other parameters
        # self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nsample = nsample
        self.npoint = npoint
        self.radius = radius
        # Define the KPConv class
        self.KPConv = KPConv(kernel_size=15,
                             p_dim=3,
                             in_channels=in_dim,
                             out_channels=out_dim,
                             KP_extent=current_extent,
                             radius=radius,
                             fixed_kernel_points='center',
                             KP_influence='linear',
                             aggregation_mode='sum',
                             deformable=False,
                             modulated=False)

        self.bns = nn.BatchNorm1d(self.out_dim, momentum=momentum)
        self.swish = Swish()
        # self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, xyz, points, grouped_xyz, neighb_idx):
        # q_pts = batch.points[self.layer_ind]
        # s_pts = batch.points[self.layer_ind]
        # neighb_inds = batch.neighbors[self.layer_ind]
        #
        # x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        # return x
        # print('-Start-' + ' ' + str(datetime.datetime.now()))

        B, N, _ = xyz.shape

        # fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint, C]
        # new_xyz = index_points(xyz, fps_idx)

        # p = torch.ones(B,N)
        # sample_idx = torch.multinomial(p, self.npoint, replacement=False)
        # new_xyz = index_points(xyz, sample_idx)

        # flag = 0
        # for i in range(B):
        #     q_pts = xyz[i,:]
        #     s_pts = xyz[i,:]
        #     neighbors = grouped_xyz[i,:]
        #     neighb_inds = neigh_idx[i,:]
        #     x = points[i,:]
        #     x = self.KPConv(q_pts, s_pts, neighbors, neighb_inds, x)
        #     x = self.bns(x)
        #     x = self.leaky_relu(x)
        #     if flag == 0:
        #         new_points = x.unsqueeze(0)
        #         flag = 1
        #     else:
        #         new_points = torch.cat((new_points,x.unsqueeze(0)),dim=0)
        new_points = self.KPConv(xyz, xyz, grouped_xyz, neighb_idx, points)
        new_points = new_points.permute(0, 2, 1).contiguous()
        new_points = self.swish(self.bns(new_points))
        new_points = new_points.permute(0, 2, 1).contiguous()
        return xyz, new_points


class Cost_volume(nn.Module):
    def __init__(self, in_channel, nsample, nsample_q, mlp1, mlp2, momentum=0.1, knn=True):
        super(Cost_volume, self).__init__()
        self.nsample_q = nsample_q
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs1 = nn.ModuleList()
        self.mlp_convs2 = nn.ModuleList()
        self.mlp_convs3 = nn.ModuleList()
        self.mlp_bns1 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()

        last_channel1 = 10 + in_channel * 2
        for out_channel in mlp1:
            self.mlp_convs1.append(nn.Conv2d(last_channel1, out_channel, 1))
            self.mlp_bns1.append(nn.BatchNorm2d(out_channel, momentum=momentum))
            last_channel1 = out_channel
        self.encode1 = nn.Conv2d(10, out_channel, 1)
        self.bn1 = nn.BatchNorm2d(out_channel, momentum=momentum)

        last_channel2 = mlp1[-1] * 2
        for out_channel in mlp2:
            self.mlp_convs2.append(nn.Conv2d(last_channel2, out_channel, 1))
            self.mlp_bns2.append(nn.BatchNorm2d(out_channel, momentum=momentum))
            last_channel2 = out_channel
        self.encode2 = nn.Conv2d(10, out_channel, 1)
        self.bn2 = nn.BatchNorm2d(out_channel, momentum=momentum)

        last_channel3 = mlp1[-1] + mlp2[-1] + in_channel
        for out_channel in mlp2:
            self.mlp_convs3.append(nn.Conv2d(last_channel3, out_channel, 1))
            last_channel3 = out_channel

    def forward(self, warped_xyz, f2_xyz, warped_points, f2_points):

        ### FIRST AGGREGATE

        _, idx_q = knn_point(self.nsample_q, f2_xyz, warped_xyz)
        qi_xyz_grouped = index_points(f2_xyz, idx_q)
        qi_points_grouped = index_points(f2_points, idx_q)

        pi_xyz_expanded = torch.repeat_interleave(torch.unsqueeze(warped_xyz, 2), repeats=self.nsample_q,
                                                  dim=2)  # batch_size, npoints, nsample, 3
        pi_points_expanded = torch.repeat_interleave(torch.unsqueeze(warped_points, 2), repeats=self.nsample_q,
                                                     dim=2)  # batch_size, npoints, nsample, 3

        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded

        pi_euc_diff = torch.sqrt(torch.sum(torch.square(pi_xyz_diff), dim=[-1], keepdim=True) + 1e-20)

        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff], dim=3)

        pi_feat_diff = torch.cat([pi_points_expanded, qi_points_grouped], dim=-1)
        pi_feat1_new = torch.cat([pi_xyz_diff_concat, pi_feat_diff],
                                 dim=3)  # batch_size, npoint*m, nsample, [channel or 1] + 3
        pi_feat1_new = pi_feat1_new.permute(0, 3, 1, 2).contiguous()  # [B, C, N, S]

        for i, conv in enumerate(self.mlp_convs1):
            bn = self.mlp_bns1[i]
            pi_feat1_new = F.relu(bn(conv(pi_feat1_new)))

        pi_xyz_diff_concat = pi_xyz_diff_concat.permute(0, 3, 1, 2).contiguous()  # [B, C, N, S]
        pi_xyz_encoding = F.relu(self.bn1(self.encode1(pi_xyz_diff_concat)))

        pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new], dim=1)

        for i, conv in enumerate(self.mlp_convs2):
            bn = self.mlp_bns2[i]
            pi_concat = F.relu(bn(conv(pi_concat)))

        WQ = F.softmax(pi_concat, dim=3)

        pi_feat1_new = WQ * pi_feat1_new.contiguous()
        pi_feat1_new = torch.sum(pi_feat1_new, dim=3, keepdim=False)  # b, n, mlp1[-1]
        pi_feat1_new = pi_feat1_new.permute(0, 2, 1).contiguous()
        ##### SECOND AGGREGATE

        _, idx = knn_point(self.nsample, warped_xyz, warped_xyz)
        pc_xyz_grouped = index_points(warped_xyz, idx)
        pc_points_grouped = index_points(pi_feat1_new, idx)

        pc_xyz_new = torch.repeat_interleave(torch.unsqueeze(warped_xyz, 2), repeats=self.nsample, dim=2)
        pc_points_new = torch.repeat_interleave(torch.unsqueeze(warped_points, 2), repeats=self.nsample, dim=2)
        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  ####b , n ,m ,3
        pc_euc_diff = torch.sqrt(torch.sum(torch.square(pc_xyz_diff), dim=3, keepdim=True) + 1e-20)
        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff], dim=3)

        pc_xyz_diff_concat = pc_xyz_diff_concat.permute(0, 3, 1, 2).contiguous()  # [B, C, N, S]
        pc_xyz_encoding = F.relu(self.bn2(self.encode2(pc_xyz_diff_concat)))

        # pc_xyz_encoding = tf_util.conv2d(pc_xyz_diff_concat, mlp1[-1], [1, 1],
        #                                  padding='VALID', stride=[1, 1],
        #                                  bn=True, is_training=is_training,
        #                                  scope='sum_xyz_encoding', bn_decay=bn_decay)

        pc_concat = torch.cat([pc_xyz_encoding.permute(0, 2, 3, 1), pc_points_new, pc_points_grouped],
                              dim=-1)  # [B, N, S, C]
        pc_concat = pc_concat.permute(0, 3, 1, 2)
        for j, conv in enumerate(self.mlp_convs3):
            bn = self.mlp_bns2[j]
            pc_concat = F.relu(bn(conv(pc_concat)))

        # for j, num_out_channel in enumerate(mlp2):
        #     pc_concat = tf_util.conv2d(pc_concat, num_out_channel, [1, 1],
        #                                padding='VALID', stride=[1, 1],
        #                                bn=True, is_training=is_training,
        #                                scope='sum_cost_volume_%d' % (j), bn_decay=bn_decay)

        WP = F.softmax(pc_concat, dim=3)  #####  b, npoints, nsample, mlp[-1]
        pc_feat1_new = WP * pc_points_grouped.permute(0, 3, 1, 2).contiguous()

        pc_feat1_new = torch.sum(pc_feat1_new, dim=3, keepdim=False)  # b*n*mlp2[-1]
        pc_feat1_new = pc_feat1_new.permute(0, 2, 1).contiguous()
        return pc_feat1_new


class flow_predictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(flow_predictor, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel, momentum=0.1))
            last_channel = out_channel

    def forward(self, points_f1, upsampled_feat, cost_volume):

        if points_f1 == None:
            points_concat = cost_volume

        elif upsampled_feat != None:
            points_concat = torch.cat([points_f1, cost_volume, upsampled_feat],
                                      dim=-1)  # B,ndataset1,nchannel1+nchannel2

        elif upsampled_feat == None:
            points_concat = torch.cat([points_f1, cost_volume], dim=-1)  # B,ndataset1,nchannel1+nchannel2

        points_concat = torch.unsqueeze(points_concat, dim=2)

        points_concat = points_concat.permute(0, 3, 1, 2).contiguous()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points_concat = F.relu(bn(conv(points_concat)))
        points_concat = points_concat.permute(0, 2, 3, 1).contiguous()

        points_concat = torch.squeeze(points_concat, dim=2)

        return points_concat

class points_filter(nn.Module):
    def __init__(self, in_channel, mlp):
        super(points_filter, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel, momentum=0.1))
            last_channel = out_channel

    def forward(self, points_f1, points_flow):


        points_concat = torch.cat([points_f1, points_flow], dim=-1)  # B,ndataset1,nchannel1+nchannel2

        points_concat = torch.unsqueeze(points_concat, dim=2)

        points_concat = points_concat.permute(0, 3, 1, 2).contiguous()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points_concat = F.relu(bn(conv(points_concat)))
        points_concat = points_concat.permute(0, 2, 3, 1).contiguous()

        points_concat = torch.squeeze(points_concat, dim=2)

        return points_concat


class FPN(nn.Module):

    def __init__(self, in_channel, out_channel, nsample):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(FPN, self).__init__()
        self.nsample = nsample
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        return

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points positi on data, [B, S, 3]
            points1: input points data, [B, N, D1]
            points2: input points data, [B, S, D2]
        Return:
            new_points: upsampled points data, [B, N, D1+D2]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        # points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        new_points = points1.permute(0, 2, 1).contiguous()
        new_points = self.conv(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()

        dists, idx = knn_point(self.nsample, xyz1, xyz2)
        # dists = square_distance(xyz1, xyz2)
        # dists, idx = dists.sort(dim=-1)
        # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, self.nsample, 1), dim=2)

        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1).contiguous()
        new_points = self.conv(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_points

def NearestUpsample(xyz1,xyz2,points,nsample):
    B, N, C = xyz1.shape
    _, S, _ = xyz2.shape
    dists, idx = knn_point(nsample, xyz2, xyz1)

    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    points = index_points(points,idx)
    interpolated_points = torch.sum(points * weight.view(B, N, nsample, 1), dim=2)
    # interpolated_points = torch.sum(index_points(points, idx), dim=2)
    return interpolated_points

def NearestUpsample_pro(xyz1,xyz2,points,nsample):
    B, N, _ = xyz1.shape
    _, S, C = points.shape
    dists, idx = knn_point(nsample, xyz1, xyz2)

    sigma2 = torch.var(dists, dim=-1, keepdim=True)
    Gauss = Gaussian_Distribution(miu=0,sigma2=sigma2)

    weight = Gauss.one_d_gaussian(dists)
    repeat_points = torch.repeat_interleave(torch.unsqueeze(points, 2), repeats=nsample, dim=2)
    weight_points = repeat_points*weight.view(B, -1, nsample, 1)
    # grouped_xyz1 = index_points(xyz1, idx)
    # points = index_points(points, idx)
    interpolated_points = torch.zeros(B, N, C).to(points.device)

    B = 1
    # for b in range(B):
    #     for s in range(S):
    #         for n in range(nsample):
    #             interpolated_points[b,idx[b,s,n],:] += weight_points[b,s,n,:]
    # for b in range(B):
    for i in range(N):
        idxx = (idx == i)
        p = weight_points * idxx.unsqueeze(-1)
        p = torch.sum(p, dim=(1, 2), keepdim=False)
        interpolated_points[:, i, :] = p

    # interpolated_points = torch.sum(points * weight.view(B, N, nsample, 1), dim=2)
    # interpolated_points = torch.sum(index_points(points, idx), dim=2)
    return interpolated_points


class Gaussian_Distribution():
    def __init__(self,miu,sigma2):
        self.miu = miu
        self.sigma2 = sigma2

    def one_d_gaussian(self,x):

        N = torch.sqrt(2*np.pi*self.sigma2)
        fac = -torch.pow(x-self.miu,2)/(self.sigma2*2)
        fac = torch.exp(fac)
        # return fac/N
        return fac

class NearestUpsampleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, nsample):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.nsample = nsample
        self.conv = nn.Conv1d(in_channel + out_channel, out_channel, 1)
        return

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, N, D1]
            points2: input points data, [B, S, D2]
        Return:
            new_points: upsampled points data, [B, N, D1+D2]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        # points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists, idx = knn_point(self.nsample, xyz1, xyz2)
            # dists = square_distance(xyz1, xyz2)
            # dists, idx = dists.sort(dim=-1)
            # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, self.nsample, 1), dim=2)

        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
            new_points = new_points.permute(0, 2, 1).contiguous()
            new_points = self.conv(new_points)
            new_points = new_points.permute(0, 2, 1).contiguous()
        else:
            new_points = interpolated_points


        return new_points


class set_upconv_module(nn.Module):
    def __init__(self, in_channel, last_channel, nsample=8, mlp1=[128, 64], mlp2=[64], pooling='max'):
        super(set_upconv_module, self).__init__()

        self.nsample = nsample
        self.pooling = pooling

        self.mlp_convs1 = nn.ModuleList()
        self.mlp_convs2 = nn.ModuleList()
        self.mlp_bns1 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()
        last_channel1 = last_channel + 3
        for out_channel in mlp1:
            self.mlp_convs1.append(nn.Conv2d(last_channel1, out_channel, 1))
            self.mlp_bns1.append(nn.BatchNorm2d(out_channel, momentum=0.1))
            last_channel1 = out_channel

        last_channel2 = mlp1[-1] + in_channel
        for out_channel in mlp2:
            self.mlp_convs2.append(nn.Conv2d(last_channel2, out_channel, 1))
            self.mlp_bns2.append(nn.BatchNorm2d(out_channel, momentum=0.1))
            last_channel2 = out_channel

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: input points position data, [B, S, 3]
            points1: input points data, [B, N, D1]
            points2: input points data, [B, S, D2]
        Return:
            new_points: sample points feature data, [B, N, D']
        """

        _, idx_q = knn_point(self.nsample, xyz2, xyz1)

        xyz2_grouped = index_points(xyz2, idx_q)
        feat2_grouped = index_points(feat2, idx_q)

        xyz1_expanded = torch.unsqueeze(xyz1, 2)  # batch_size, npoint1, 1, 3
        xyz_diff = xyz2_grouped - xyz1_expanded  # batch_size, npoint1, nsample, 3

        net = torch.cat([feat2_grouped, xyz_diff], dim=3)  # batch_size, npoint1, nsample, channel2+3

        net = net.permute(0, 3, 1, 2).contiguous()  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs1):
            bn = self.mlp_bns1[i]
            net = F.relu(bn(conv(net)))
        net = net.permute(0, 2, 3, 1).contiguous()

        if self.pooling == 'max':
            feat1_new = torch.max(net, dim=2, keepdim=False).values  # batch_size, npoint1, mlp[-1]
        elif self.pooling == 'avg':
            feat1_new = torch.mean(net, dim=2, keepdim=False)  # batch_size, npoint1, mlp[-1]

        if feat1 is not None:
            feat1_new = torch.cat([feat1_new, feat1], dim=2)  # batch_size, npoint1, mlp[-1]+channel1

        feat1_new = torch.unsqueeze(feat1_new, 2)  # batch_size, npoint1, 1, mlp[-1]+channel2

        feat1_new = feat1_new.permute(0, 3, 1, 2).contiguous()  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs2):
            bn = self.mlp_bns2[i]
            feat1_new = F.relu(bn(conv(feat1_new)))
        feat1_new = feat1_new.permute(0, 2, 3, 1).contiguous()

        feat1_new = torch.squeeze(feat1_new, 2)  # batch_size, npoint1, mlp2[-1]
        return feat1_new


class RefineBlock(nn.Module):
    def __init__(self, in_channel, last_channel, npoint=256, nsample=4, nsample_q=6, mlp1=[128, 64, 64], mlp2=[128, 64]):
        super().__init__()

        self.npoint = npoint
        # self.cost_volume = Cost_volume(in_channel=in_channel, nsample=nsample, nsample_q=nsample_q, mlp1=mlp1,
        #                                mlp2=mlp2)
        self.fe_layer = FlowEmbedding(in_channel=in_channel, nsample=nsample, nsample_q=nsample_q, mlp1=mlp1, mlp2=mlp2,
                                      radius=4.0, pooling='sum', corr_func='concat')

        self.upconv = set_upconv_module(in_channel=in_channel, last_channel=last_channel, nsample=8, mlp1=[128, 64], mlp2=[64])

        # self.MBConv = MBConv3D(in_channel=in_channel*2, out_channel=in_channel, npoint=npoint, nsample=nsample, radius=radius, expand_ratio=4)
        self.mlp = SharedMLPBlock(in_channel=128,out_channel=64,hs_swish=True,bias=True)
        self.filter = points_filter(in_channel=in_channel+64, mlp=[128, 64])
        # last_channel = in_channel * 2
        # for out_channel in mlp2:
        #     self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #     self.mlp_bns.append(nn.BatchNorm2d(out_channel, momentum=0.1))
        #     last_channel = out_channel
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.5)

        self.linear1 = nn.Linear(mlp2[-1], 256)
        self.linear2 = nn.Linear(256, 4)
        self.linear3 = nn.Linear(256, 3)


    def forward(self, pos1, pos2, pos1_l, feature1, feature2, pre_q, pre_t, points_predict):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        batch_size = pos1.shape[0]
        q_coarse = torch.reshape(pre_q, [batch_size, 1, -1])
        t_coarse = torch.reshape(pre_t, [batch_size, 1, -1])
        q_inv = inv_q(q_coarse)

        # warped flow
        pc1_sample_q = torch.cat((torch.zeros([batch_size, self.npoint, 1]).to(pos1.device), pos1), dim=-1)
        xyz_warped = mul_q_point(q_coarse, pc1_sample_q, batch_size, True)
        pos1_warped = mul_q_point(xyz_warped, q_inv, batch_size, False)[:, :, 1:] + t_coarse

        # cost volume
        # l3_points_f1, l3_points_f2 = self.attn(l3_points_f1, l3_points_f2)
        _, flow_feature = self.fe_layer(pos1_warped, pos2, feature1, feature2)

        # flow_feature = self.cost_volume(pos1_warped, pos2, feature1, feature2)
        flow_feature_l_up = self.upconv(pos1, pos1_l, feature1, points_predict)

        flow_feature_cat = torch.cat([flow_feature, flow_feature_l_up], dim=-1)

        _, flow_feature_new = self.mlp(pos1_warped, flow_feature_cat)
        W_flow = self.filter(feature1,flow_feature_new)
        W_flow_soft = F.softmax(W_flow, dim=1)
        flow_sum = torch.sum(flow_feature_new * W_flow_soft, dim=1, keepdim=True)

        # _, flow_feature_new = self.MBConv(pos1_warped, flow_feature_new)
        # for i, conv in enumerate(self.mlp_convs):
        #     bn = self.mlp_bns[i]
        #     flow_feature_new = F.relu(bn(conv(flow_feature_new)))
        # flow_feature_pose = self.avg_pooling(flow_feature_new).squeeze(-1)
        # flow_feature_pose = torch.max(flow_feature_new, dim=1, keepdim=False).values
        # flow_feature_pose = torch.sum(flow_feature_new, dim=1, keepdim=False)
        flow_feature_pose = self.linear1(flow_sum)
        feature_q = self.drop(flow_feature_pose)
        feature_t = self.drop(flow_feature_pose)

        new_q_det = self.linear2(feature_q)
        new_t_det = self.linear3(feature_t)

        new_q_det = new_q_det / (torch.sqrt(torch.sum(new_q_det * new_q_det, dim=-1, keepdim=True)
                                                           + torch.Tensor([1e-10]).to(new_q_det.device)) + 1e-10)
        new_t_coarse_trans = torch.cat((torch.zeros([batch_size, 1, 1]).to(pos1.device), t_coarse), dim=-1)
        new_t_coarse_trans = mul_q_point(q_coarse, new_t_coarse_trans, batch_size, True)
        new_t_coarse_trans = mul_q_point(new_t_coarse_trans, q_inv, batch_size, False)[:, :, 1:]  #### q t_coarse q_1

        new_q = torch.squeeze(mul_q_point(new_q_det, q_coarse, batch_size, False),1)
        new_t = torch.squeeze(new_t_coarse_trans + new_t_det,1)

        return new_q, new_t, flow_feature_new, W_flow

class RefineBlock_withoutmask(nn.Module):
    def __init__(self, in_channel, last_channel, npoint=256, nsample=4, nsample_q=6, mlp1=[128, 64, 64], mlp2=[128, 64]):
        super().__init__()

        self.npoint = npoint
        # self.cost_volume = Cost_volume(in_channel=in_channel, nsample=nsample, nsample_q=nsample_q, mlp1=mlp1,
        #                                mlp2=mlp2)
        self.fe_layer = FlowEmbedding(in_channel=in_channel, nsample=nsample, nsample_q=nsample_q, mlp1=mlp1, mlp2=mlp2,
                                      radius=4.0, pooling='sum', corr_func='concat')

        self.upconv = set_upconv_module(in_channel=in_channel, last_channel=64, nsample=8, mlp1=[128, 64], mlp2=[64])

        # self.MBConv = MBConv3D(in_channel=in_channel*2, out_channel=in_channel, npoint=npoint, nsample=nsample, radius=radius, expand_ratio=4)
        self.mlp = SharedMLPBlock(in_channel=128, out_channel=64, hs_swish=True, bias=True)
        self.filter = points_filter(in_channel=in_channel + 64, mlp=[128, 64])
        # last_channel = in_channel * 2
        # for out_channel in mlp2:
        #     self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #     self.mlp_bns.append(nn.BatchNorm2d(out_channel, momentum=0.1))
        #     last_channel = out_channel
        # self.bn = nn.BatchNorm1d(64)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.5)

        self.linear1 = nn.Linear(mlp2[-1], 256)
        self.linear2 = nn.Linear(256, 4)
        self.linear3 = nn.Linear(256, 3)

    def forward(self, pos1, pos2, pos1_l, feature1, feature2, pre_q, pre_t, points_predict):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        batch_size = pos1.shape[0]
        q_coarse = torch.reshape(pre_q, [batch_size, 1, -1])
        t_coarse = torch.reshape(pre_t, [batch_size, 1, -1])
        q_inv = inv_q(q_coarse)

        # warped flow
        pc1_sample_q = torch.cat((torch.zeros([batch_size, self.npoint, 1]).to(pos1.device), pos1), dim=-1)
        xyz_warped = mul_q_point(q_coarse, pc1_sample_q, batch_size, True)
        pos1_warped = mul_q_point(xyz_warped, q_inv, batch_size, False)[:, :, 1:] + t_coarse

        # cost volume
        # l3_points_f1, l3_points_f2 = self.attn(l3_points_f1, l3_points_f2)
        _, flow_feature = self.fe_layer(pos1_warped, pos2, feature1, feature2)

        # flow_feature = self.cost_volume(pos1_warped, pos2, feature1, feature2)
        flow_feature_l_up = self.upconv(pos1, pos1_l, feature1, points_predict)

        flow_feature_cat = torch.cat([flow_feature, flow_feature_l_up], dim=-1)

        _, flow_feature_new = self.mlp(pos1_warped, flow_feature_cat)

        flow_feature_predict = self.avg_pooling(flow_feature_new.permute(0, 2, 1).contiguous())
        flow_feature_predict = flow_feature_predict.permute(0, 2, 1).contiguous()

        # flow_sum = torch.sum(flow_feature_predict, dim=1, keepdim=True)
        # flow_sum = torch.sum(flow_feature_new, dim=1, keepdim=True)
        flow_feature_pose = self.linear1(flow_feature_predict)
        feature_q = self.drop(flow_feature_pose)
        feature_t = self.drop(flow_feature_pose)

        new_q_det = self.linear2(feature_q)
        new_t_det = self.linear3(feature_t)

        new_q_det = new_q_det / (torch.sqrt(torch.sum(new_q_det * new_q_det, dim=-1, keepdim=True)
                                            + torch.Tensor([1e-10]).to(new_q_det.device)) + 1e-10)
        new_t_coarse_trans = torch.cat((torch.zeros([batch_size, 1, 1]).to(pos1.device), t_coarse), dim=-1)
        new_t_coarse_trans = mul_q_point(q_coarse, new_t_coarse_trans, batch_size, True)
        new_t_coarse_trans = mul_q_point(new_t_coarse_trans, q_inv, batch_size, False)[:, :, 1:]  #### q t_coarse q_1

        new_q = torch.squeeze(mul_q_point(new_q_det, q_coarse, batch_size, False), 1)
        new_t = torch.squeeze(new_t_coarse_trans + new_t_det, 1)

        return new_q, new_t, flow_feature_new


class RefineBlock_PWCLO(nn.Module):
    def __init__(self, in_channel, npoint=256, nsample=16, nsample_q=32, mlp1=[128, 64, 64], mlp2=[128, 64]):
        super().__init__()

        self.npoint = npoint
        # Expansion phase (Inverted Bottleneck)

        self.cost_volume = Cost_volume(in_channel=in_channel, nsample=nsample, nsample_q=nsample_q, mlp1=mlp1,
                                       mlp2=mlp2)

        self.upconv1 = set_upconv_module(in_channel=in_channel, last_channel=64, nsample=8, mlp1=[128, 64], mlp2=[64])
        self.upconv2 = set_upconv_module(in_channel=in_channel, last_channel=64, nsample=8, mlp1=[128, 64], mlp2=[64])
        self.flow1 = flow_predictor(in_channel=in_channel + 64 * 2, mlp=[128, 64])
        self.flow2 = flow_predictor(in_channel=in_channel + 64 * 2, mlp=[128, 64])

        self.drop = nn.Dropout(0.5)
        self.line1 = nn.Linear(64, 256)
        self.line2 = nn.Linear(256, 4)
        self.line3 = nn.Linear(256, 3)
        self.swish = Swish()

    def forward(self, pos1, pos2, pos1_l, feature1, feature2, pre_q, pre_t, cost_volume_w, points_predict):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        batch_size = pos1.shape[0]
        q_coarse = torch.reshape(pre_q, [batch_size, 1, -1])
        t_coarse = torch.reshape(pre_t, [batch_size, 1, -1])
        q_inv = inv_q(q_coarse)

        # warped flow
        pc1_sample_q = torch.cat((torch.zeros([batch_size, self.npoint, 1]).to(pos1.device), pos1), dim=-1)
        pos1_warped = mul_q_point(q_coarse, pc1_sample_q, batch_size, True)
        pos1_warped = mul_q_point(pos1_warped, q_inv, batch_size, False)[:, :, 1:] + t_coarse

        cost_volume = self.cost_volume(pos1_warped, pos2, feature1, feature2)

        cost_volume_w_upsample = self.upconv1(pos1, pos1_l, feature1, cost_volume_w)
        cost_volume_upsample = self.upconv2(pos1, pos1_l, feature1, points_predict)

        cost_volume_predict = self.flow1(feature1, cost_volume_upsample, cost_volume)
        cost_volume_w = self.flow2(cost_volume_w_upsample, feature1, cost_volume_predict)

        W_cost_volume = F.softmax(cost_volume_w, dim=1)

        # Feature Propagation
        flow_feature_new = torch.sum(cost_volume_predict * W_cost_volume, dim=1, keepdim=True)
        flow_feature_pose = self.line1(flow_feature_new)
        feature_q = self.drop(flow_feature_pose)
        feature_t = self.drop(flow_feature_pose)

        new_q_det = self.line2(feature_q)
        new_q_det = new_q_det / (
                torch.sqrt(torch.sum(new_q_det * new_q_det, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        new_t_det = self.line3(feature_t)

        new_t_coarse_trans = torch.cat((torch.zeros([batch_size, 1, 1]).to(pos1.device), t_coarse), dim=-1)
        new_t_coarse_trans = mul_q_point(q_coarse, new_t_coarse_trans, batch_size, True)
        new_t_coarse_trans = mul_q_point(new_t_coarse_trans, q_inv, batch_size, False)[:, :, 1:]  #### q t_coarse q_1

        new_q = torch.squeeze(mul_q_point(new_q_det, q_coarse, batch_size, False))
        new_t = torch.squeeze(new_t_coarse_trans + new_t_det)

        return new_q, new_t, cost_volume_w, cost_volume_predict


class GraphAttention(nn.Module):
    def __init__(self, all_channel, feature_dim, dropout, alpha):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        '''
        Input:
            center_xyz: sampled points position data [B, npoint, C]
            center_feature: centered point feature [B, npoint, D]
            grouped_xyz: group xyz data [B, npoint, nsample, C]
            grouped_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            graph_pooling: results of graph pooling [B, npoint, D]
        '''
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        delta_p = center_xyz.contiguous().view(B, npoint, 1, C).expand(B, npoint, nsample,
                                                                       C) - grouped_xyz  # [B, npoint, nsample, C]
        delta_h = center_feature.contiguous().view(B, npoint, 1, D).expand(B, npoint, nsample,
                                                                           D) - grouped_feature  # [B, npoint, nsample, D]
        delta_p_concat_h = torch.cat([delta_p, delta_h], dim=-1)  # [B, npoint, nsample, C+D]
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a))  # [B, npoint, nsample,D]
        attention = F.softmax(e, dim=2)  # [B, npoint, nsample,D]
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature), dim=2)  # [B, npoint, D]
        return graph_pooling


class GraphAttentionConvLayer(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, droupout=0.6, alpha=0.2):
        super(GraphAttentionConvLayer, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.droupout = droupout
        self.alpha = alpha
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.GAT = GraphAttention(3 + last_channel, last_channel, self.droupout, self.alpha)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # xyz = xyz.permute(0, 2, 1)
        # if points is not None:
        #     points = points.permute(0, 2, 1)

        new_xyz, new_points, grouped_xyz, fps_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz,
                                                                        points, True)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # fps_points: [B, npoint, C+D,1]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # [B, C+D, nsample,npoint]
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1).contiguous()  # [B, C+D, 1,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fps_points = F.relu(bn(conv(fps_points)))
            new_points = F.relu(bn(conv(new_points)))
        # new_points: [B, F, nsample,npoint]
        # fps_points: [B, F, 1,npoint]
        fps_points = fps_points.squeeze(2).permute(0, 2, 1).contiguous()
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.GAT(center_xyz=new_xyz,
                              center_feature=fps_points,
                              grouped_xyz=grouped_xyz,
                              grouped_feature=new_points)
        # new_xyz = new_xyz.permute(0, 2, 1)
        # new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points
