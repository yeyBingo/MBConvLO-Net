from models.pointnet2_util import *
import time

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.name = "MBConvLO-Net"
        self.args = args

        self.MBConv1 = MBConv3D_fps(in_channel=4, out_channel=16, momentum=0.1, npoint=2048, nsample=32, radius=1,
                                expand_ratio=1)
        self.MBConv2 = MBConv3D_fps(in_channel=16, out_channel=32, momentum=0.1, npoint=1024, nsample=32, radius=2,
                                expand_ratio=4)
        self.MBConv3 = MBConv3D_fps(in_channel=32, out_channel=64, momentum=0.1, npoint=256, nsample=16, radius=2,
                                expand_ratio=4)

        self.fe_layer = FlowEmbedding(in_channel=64, nsample=4, nsample_q=16, mlp1=[128, 64, 64], mlp2=[128, 64],
                                      pooling='sum', corr_func='concat')

        self.filter = points_filter(in_channel=64 + 64, mlp=[128, 64])

        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.5)
        self.linear1 = nn.Linear(64, 256)
        self.linear2 = nn.Linear(256, 4)
        self.linear3 = nn.Linear(256, 3)


        self.refine1 = RefineBlock(in_channel=32, last_channel=64, npoint=1024, nsample=4, nsample_q=6,
                                   mlp1=[128, 64, 64], mlp2=[128, 64])
        self.refine2 = RefineBlock(in_channel=16, last_channel=64, npoint=2048, nsample=4, nsample_q=6,
                                   mlp1=[128, 64, 64], mlp2=[128, 64])

        self.n = 0
    def forward(self, xyz1, xyz2):
        # num_point = xyz1.size(1)# // 2

        l0_xyz_f1 = xyz1[:, :, :3]
        l0_points_f1 = xyz1

        l0_xyz_f2 = xyz2[:, :, :3]
        l0_points_f2 = xyz2

        l1_xyz_f1, l1_points_f1 = self.MBConv1(l0_xyz_f1, l0_points_f1)
        l2_xyz_f1, l2_points_f1 = self.MBConv2(l1_xyz_f1, l1_points_f1)
        l3_xyz_f1, l3_points_f1 = self.MBConv3(l2_xyz_f1, l2_points_f1)


        l1_xyz_f2, l1_points_f2 = self.MBConv1(l0_xyz_f2, l0_points_f2)
        l2_xyz_f2, l2_points_f2 = self.MBConv2(l1_xyz_f2, l1_points_f2)
        l3_xyz_f2, l3_points_f2 = self.MBConv3(l2_xyz_f2, l2_points_f2)


        _, l3_points_flow = self.fe_layer(l3_xyz_f1, l3_xyz_f2, l3_points_f1, l3_points_f2)
        l3_points_predict = l3_points_flow
        W_l3_flow = self.filter(l3_points_f1, l3_points_flow)
        W_l3_flow_soft = F.softmax(W_l3_flow, dim=1)
        l3_flow_sum = torch.sum(l3_points_flow * W_l3_flow_soft, dim=1, keepdim=False)

        l3_points_f1_pose = self.linear1(l3_flow_sum)
        l3_points_f1_q = self.drop(l3_points_f1_pose)
        l3_points_f1_t = self.drop(l3_points_f1_pose)

        l3_q_coarse = self.linear2(l3_points_f1_q)
        l3_t_coarse = self.linear3(l3_points_f1_t)

        l3_q = torch.squeeze(l3_q_coarse, 1)
        l3_t = torch.squeeze(l3_t_coarse, 1)


        l2_q, l2_t, l2_points_predict, W_l2_flow = self.refine1(l2_xyz_f1, l2_xyz_f2, l3_xyz_f1,
                                                                           l2_points_f1, l2_points_f2, l3_q, l3_t,
                                                                           l3_points_predict)
        l1_q, l1_t, l1_points_predict, W_l1_flow = self.refine2(l1_xyz_f1, l1_xyz_f2, l2_xyz_f1,
                                                                           l1_points_f1, l1_points_f2, l2_q, l2_t,
                                                                           l2_points_predict)

        l3_q_norm = l3_q / (torch.sqrt(torch.sum(l3_q * l3_q, dim=-1, keepdim=True)
                                       + torch.Tensor([1e-10]).to(l3_q.device)) + 1e-10)
        l2_q_norm = l2_q / (torch.sqrt(torch.sum(l2_q * l2_q, dim=-1, keepdim=True)
                                       + torch.Tensor([1e-10]).to(l2_q.device)) + 1e-10)
        l1_q_norm = l1_q / (torch.sqrt(torch.sum(l1_q * l1_q, dim=-1, keepdim=True)
                                       + torch.Tensor([1e-10]).to(l1_q.device)) + 1e-10)

        return l3_q_norm, l3_t, l2_q_norm, l2_t, l1_q_norm, l1_t




class get_loss(nn.Module):
    def __init__(self, w_x, w_q):
        super(get_loss, self).__init__()
        self.w_x = nn.Parameter(w_x)
        self.w_q = nn.Parameter(w_q)

        # self.w_x = w_x
        # self.w_q = w_q

    def forward(self, outputs, q_gt, t_gt):  #####idx来选择真值

        t_gt = torch.squeeze(t_gt)  ###  8,3
        # l4_q, l4_t, l3_q, l3_t, l2_q, l2_t, l1_q, l1_t = outputs
        l3_q, l3_t, l2_q, l2_t, l1_q, l1_t = outputs
        # l4_q, l4_t= outputs

        # l4_q_norm = l4_q / (torch.sqrt(torch.sum(l4_q * l4_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        # l4_loss_q = torch.mean(
        #     torch.sqrt(torch.sum((q_gt - l4_q_norm) * (q_gt - l4_q_norm), dim=-1, keepdim=True) + 1e-10))
        # l4_loss_x = torch.mean(torch.sqrt((l4_t - t_gt) * (l4_t - t_gt) + 1e-10))

        l3_q_norm = l3_q / (torch.sqrt(torch.sum(l3_q * l3_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l3_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt - l3_q_norm) * (q_gt - l3_q_norm), dim=-1, keepdim=True) + 1e-10))
        l3_loss_x = torch.mean(torch.sqrt((l3_t - t_gt) * (l3_t - t_gt) + 1e-10))

        l2_q_norm = l2_q / (torch.sqrt(torch.sum(l2_q * l2_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l2_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt - l2_q_norm) * (q_gt - l2_q_norm), dim=-1, keepdim=True) + 1e-10))
        l2_loss_x = torch.mean(torch.sqrt((l2_t - t_gt) * (l2_t - t_gt) + 1e-10))

        l1_q_norm = l1_q / (torch.sqrt(torch.sum(l1_q * l1_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l1_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt - l1_q_norm) * (q_gt - l1_q_norm), dim=-1, keepdim=True) + 1e-10))
        l1_loss_x = torch.mean(torch.sqrt((l1_t - t_gt) * (l1_t - t_gt) + 1e-10))

        # l4_loss = l4_loss_x * torch.exp(-self.w_x) + self.w_x + l4_loss_q * torch.exp(-self.w_q) + self.w_q
        l3_loss = l3_loss_x * torch.exp(-self.w_x) + self.w_x + l3_loss_q * torch.exp(-self.w_q) + self.w_q
        l2_loss = l2_loss_x * torch.exp(-self.w_x) + self.w_x + l2_loss_q * torch.exp(-self.w_q) + self.w_q
        l1_loss = l1_loss_x * torch.exp(-self.w_x) + self.w_x + l1_loss_q * torch.exp(-self.w_q) + self.w_q

        # loss_sum = l4_loss + l3_loss + l2_loss + l1_loss
        # print('l4:%.4f l3:%.4f l2:%.4f l1:%.4f' % (l4_loss, l3_loss, l2_loss, l1_loss))
        # loss_sum = 1.6 * l4_loss + 0.8 * l3_loss + 0.4 * l2_loss + 0.2 * l1_loss
        loss_sum = 1.6 * l3_loss + 0.8 * l2_loss + 0.4 * l1_loss
        # loss_sum = l4_loss

        return loss_sum

class get_loss2(nn.Module):
    def __init__(self, w_x, w_q):
        super(get_loss2, self).__init__()
        self.w_x = nn.Parameter(w_x)
        self.w_q = nn.Parameter(w_q)

        # self.w_x = w_x
        # self.w_q = w_q

    def forward(self, outputs, q_gt, t_gt):  #####idx来选择真值

        t_gt = torch.squeeze(t_gt)  ###  8,3
        l4_q, l4_t, l3_q, l3_t, l2_q, l2_t, l1_q, l1_t = outputs

        q_gt_1 = q_gt[:, 0]
        q_gt_2 = q_gt[:, 1:] / (torch.sqrt(torch.sum(q_gt[:, 1:] * q_gt[:, 1:] + 1e-10, dim=-1, keepdim=True)) + 1e-10)

        l4_q_norm = l4_q / (torch.sqrt(torch.sum(l4_q * l4_q + 1e-10, dim=-1, keepdim=True)) + 1e-10)
        l4_q_1 = l4_q_norm[:, 0]
        l4_q_2 = l4_q_norm[:, 1:] / (
                torch.sqrt(torch.sum(l4_q_norm[:, 1:] * l4_q_norm[:, 1:] + 1e-10, dim=-1, keepdim=True)) + 1e-10)
        l4_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt_1 - l4_q_1) * (q_gt_1 - l4_q_1) + 1e-10, dim=-1, keepdim=True)) +
            torch.sqrt(torch.sum((q_gt_2 - l4_q_2) * (q_gt_2 - l4_q_2) + 1e-10, dim=-1, keepdim=True))
        )
        l4_loss_x = torch.mean(torch.sqrt((l4_t - t_gt) * (l4_t - t_gt) + 1e-10))

        l3_q_norm = l3_q / (torch.sqrt(torch.sum(l3_q * l3_q + 1e-10, dim=-1, keepdim=True)) + 1e-10)
        l3_q_1 = l3_q_norm[:, 0]
        l3_q_2 = l3_q_norm[:, 1:] / (
                torch.sqrt(torch.sum(l3_q_norm[:, 1:] * l3_q_norm[:, 1:] + 1e-10, dim=-1, keepdim=True)) + 1e-10)
        l3_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt_1 - l3_q_1) * (q_gt_1 - l3_q_1) + 1e-10, dim=-1, keepdim=True)) +
            torch.sqrt(torch.sum((q_gt_2 - l3_q_2) * (q_gt_2 - l3_q_2) + 1e-10, dim=-1, keepdim=True))
        )
        l3_loss_x = torch.mean(torch.sqrt((l3_t - t_gt) * (l3_t - t_gt) + 1e-10))

        l2_q_norm = l2_q / (torch.sqrt(torch.sum(l2_q * l2_q + 1e-10, dim=-1, keepdim=True)) + 1e-10)
        l2_q_1 = l2_q_norm[:, 0]
        l2_q_2 = l2_q_norm[:, 1:] / (
                torch.sqrt(torch.sum(l2_q_norm[:, 1:] * l2_q_norm[:, 1:] + 1e-10, dim=-1, keepdim=True)) + 1e-10)
        l2_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt_1 - l2_q_1) * (q_gt_1 - l2_q_1) + 1e-10, dim=-1, keepdim=True)) +
            torch.sqrt(torch.sum((q_gt_2 - l2_q_2) * (q_gt_2 - l2_q_2) + 1e-10, dim=-1, keepdim=True))
        )
        l2_loss_x = torch.mean(torch.sqrt((l2_t - t_gt) * (l2_t - t_gt) + 1e-10))

        l1_q_norm = l1_q / (torch.sqrt(torch.sum(l1_q * l1_q + 1e-10, dim=-1, keepdim=True)) + 1e-10)
        l1_q_1 = l1_q_norm[:, 0]
        l1_q_2 = l1_q_norm[:, 1:] / (
                torch.sqrt(torch.sum(l1_q_norm[:, 1:] * l1_q_norm[:, 1:] + 1e-10, dim=-1, keepdim=True)) + 1e-10)
        l1_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt_1 - l1_q_1) * (q_gt_1 - l1_q_1) + 1e-10, dim=-1, keepdim=True)) +
            torch.sqrt(torch.sum((q_gt_2 - l1_q_2) * (q_gt_2 - l1_q_2) + 1e-10, dim=-1, keepdim=True))
        )
        l1_loss_x = torch.mean(torch.sqrt((l1_t - t_gt) * (l1_t - t_gt) + 1e-10))

        l4_loss = l4_loss_x * torch.exp(-self.w_x) + self.w_x + l4_loss_q * torch.exp(-self.w_q) + self.w_q
        l3_loss = l3_loss_x * torch.exp(-self.w_x) + self.w_x + l3_loss_q * torch.exp(-self.w_q) + self.w_q
        l2_loss = l2_loss_x * torch.exp(-self.w_x) + self.w_x + l2_loss_q * torch.exp(-self.w_q) + self.w_q
        l1_loss = l1_loss_x * torch.exp(-self.w_x) + self.w_x + l1_loss_q * torch.exp(-self.w_q) + self.w_q

        # l4_loss = l4_loss_x * torch.exp(-self.w_x) + l4_loss_q * torch.exp(-self.w_q)
        # l3_loss = l3_loss_x * torch.exp(-self.w_x) + l3_loss_q * torch.exp(-self.w_q)
        # l2_loss = l2_loss_x * torch.exp(-self.w_x) + l2_loss_q * torch.exp(-self.w_q)
        # l1_loss = l1_loss_x * torch.exp(-self.w_x) + l1_loss_q * torch.exp(-self.w_q)

        # loss_sum = l4_loss + l3_loss + l2_loss + l1_loss
        # print('l4:%.4f l3:%.4f l2:%.4f l1:%.4f' % (l4_loss, l3_loss, l2_loss, l1_loss))
        loss_sum = 1.6 * l4_loss + 0.8 * l3_loss + 0.4 * l2_loss + 0.2 * l1_loss
        # loss_sum = 0.2*l4_loss + 0.4*l3_loss + 0.8*l2_loss + 1.6*l1_loss
        # loss_sum = l4_loss

        return loss_sum

class get_loss3(nn.Module):
    def __init__(self, w_x, w_q):
        super(get_loss3, self).__init__()
        self.w_x = nn.Parameter(w_x)
        self.w_q = nn.Parameter(w_q)

    def forward(self, outputs, q_gt, t_gt):  #####idx来选择真值

        t_gt = torch.squeeze(t_gt)  ###  8,3
        # l4_q, l4_t, l3_q, l3_t, l2_q, l2_t, l1_q, l1_t = outputs
        l3_q, l3_t, l2_q, l2_t, l1_q, l1_t = outputs
        # _, _, _, _, _, _, l1_q, l1_t = outputs
        # l4_q, l4_t = outputs

        # l4_q_norm = l4_q / (torch.sqrt(torch.sum(l4_q * l4_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        # l4_loss_q = torch.mean(
        #     torch.sqrt(torch.sum((q_gt - l4_q_norm) * (q_gt - l4_q_norm), dim=-1, keepdim=True) + 1e-10))
        # l4_loss_x = torch.mean(torch.sqrt((l4_t - t_gt) * (l4_t - t_gt) + 1e-10))
        #
        # l4_loss = l4_loss_x * torch.exp(-self.w_x) + self.w_x + l4_loss_q * torch.exp(-self.w_q) + self.w_q

        l3_q_norm = l3_q / (torch.sqrt(torch.sum(l3_q * l3_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l3_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt - l3_q_norm) * (q_gt - l3_q_norm), dim=-1, keepdim=True) + 1e-10))
        l3_loss_x = torch.mean(torch.sqrt((l3_t - t_gt) * (l3_t - t_gt) + 1e-10))

        l3_loss = l3_loss_x * torch.exp(-self.w_x) + self.w_x + l3_loss_q * torch.exp(-self.w_q) + self.w_q

        l2_q_norm = l2_q / (torch.sqrt(torch.sum(l2_q * l2_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l2_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt - l2_q_norm) * (q_gt - l2_q_norm), dim=-1, keepdim=True) + 1e-10))
        l2_loss_x = torch.mean(torch.sqrt((l2_t - t_gt) * (l2_t - t_gt) + 1e-10))

        l2_loss = l2_loss_x * torch.exp(-self.w_x) + self.w_x + l2_loss_q * torch.exp(-self.w_q) + self.w_q

        l1_q_norm = l1_q / (torch.sqrt(torch.sum(l1_q * l1_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l1_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt - l1_q_norm) * (q_gt - l1_q_norm), dim=-1, keepdim=True) + 1e-10))
        l1_loss_x = torch.mean(torch.sqrt((l1_t - t_gt) * (l1_t - t_gt) + 1e-10))

        l1_loss = l1_loss_x * torch.exp(-self.w_x) + self.w_x + l1_loss_q * torch.exp(-self.w_q) + self.w_q

        # loss_sum = l4_loss + l3_loss + l2_loss + l1_loss
        # print('l4:%.4f l3:%.4f l2:%.4f l1:%.4f' % (l4_loss, l3_loss, l2_loss, l1_loss))
        # loss_sum = 1.6 * l4_loss + 0.8 * l3_loss + 0.4 * l2_loss + 0.2 * l1_loss
        loss_sum =  1.6 * l3_loss + 0.8 * l2_loss + 0.4 * l1_loss
        # loss_sum = 0.2*l4_loss + 0.4*l3_loss + 0.8*l2_loss + 1.6*l1_loss
        # loss_sum = l4_loss

        return loss_sum

class get_loss_single(nn.Module):
    def __init__(self, w_x, w_q):
        super(get_loss_single, self).__init__()
        self.w_x = nn.Parameter(w_x)
        self.w_q = nn.Parameter(w_q)

    def forward(self, outputs, q_gt, t_gt):  #####idx来选择真值

        t_gt = torch.squeeze(t_gt)  ###  8,3
        # l4_q, l4_t, l3_q, l3_t, l2_q, l2_t, l1_q, l1_t = outputs
        l3_q, l3_t = outputs
        # _, _, _, _, _, _, l1_q, l1_t = outputs
        # l4_q, l4_t = outputs

        # l4_q_norm = l4_q / (torch.sqrt(torch.sum(l4_q * l4_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        # l4_loss_q = torch.mean(
        #     torch.sqrt(torch.sum((q_gt - l4_q_norm) * (q_gt - l4_q_norm), dim=-1, keepdim=True) + 1e-10))
        # l4_loss_x = torch.mean(torch.sqrt((l4_t - t_gt) * (l4_t - t_gt) + 1e-10))
        #
        # l4_loss = l4_loss_x * torch.exp(-self.w_x) + self.w_x + l4_loss_q * torch.exp(-self.w_q) + self.w_q

        l3_q_norm = l3_q / (torch.sqrt(torch.sum(l3_q * l3_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l3_loss_q = torch.mean(
            torch.sqrt(torch.sum((q_gt - l3_q_norm) * (q_gt - l3_q_norm), dim=-1, keepdim=True) + 1e-10))
        l3_loss_x = torch.mean(torch.sqrt((l3_t - t_gt) * (l3_t - t_gt) + 1e-10))

        l3_loss = l3_loss_x * torch.exp(-self.w_x) + self.w_x + l3_loss_q * torch.exp(-self.w_q) + self.w_q
        #
        # l2_q_norm = l2_q / (torch.sqrt(torch.sum(l2_q * l2_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        # l2_loss_q = torch.mean(
        #     torch.sqrt(torch.sum((q_gt - l2_q_norm) * (q_gt - l2_q_norm), dim=-1, keepdim=True) + 1e-10))
        # l2_loss_x = torch.mean(torch.sqrt((l2_t - t_gt) * (l2_t - t_gt) + 1e-10))
        #
        # l2_loss = l2_loss_x * torch.exp(-self.w_x) + self.w_x + l2_loss_q * torch.exp(-self.w_q) + self.w_q
        #
        # l1_q_norm = l1_q / (torch.sqrt(torch.sum(l1_q * l1_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        # l1_loss_q = torch.mean(
        #     torch.sqrt(torch.sum((q_gt - l1_q_norm) * (q_gt - l1_q_norm), dim=-1, keepdim=True) + 1e-10))
        # l1_loss_x = torch.mean(torch.sqrt((l1_t - t_gt) * (l1_t - t_gt) + 1e-10))
        #
        # l1_loss = l1_loss_x * torch.exp(-self.w_x) + self.w_x + l1_loss_q * torch.exp(-self.w_q) + self.w_q

        # loss_sum = l4_loss + l3_loss + l2_loss + l1_loss
        # print('l4:%.4f l3:%.4f l2:%.4f l1:%.4f' % (l4_loss, l3_loss, l2_loss, l1_loss))
        # loss_sum = 1.6 * l4_loss + 0.8 * l3_loss + 0.4 * l2_loss + 0.2 * l1_loss
        # loss_sum =  1.6 * l3_loss + 0.8 * l2_loss + 0.4 * l1_loss
        # loss_sum = 0.2*l4_loss + 0.4*l3_loss + 0.8*l2_loss + 1.6*l1_loss
        loss_sum = l3_loss

        return loss_sum

class get_pwclo(nn.Module):
    def __init__(self, args):
        super(get_pwclo, self).__init__()
        self.name = "pwclo"
        self.args = args
        self.sa1 = PointNetSetAbstraction(npoint=2048, radius=0.1, nsample=32, in_channel=3 + 3, mlp=[8, 8, 16],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=16 + 3, mlp=[16, 16, 32],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=16, in_channel=32 + 3, mlp=[32, 32, 64],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=16, in_channel=64 + 3, mlp=[64, 64, 128],
                                          group_all=False)

        self.cost_volume = Cost_volume(in_channel=64, nsample=4, nsample_q=32, mlp1=[128, 64, 64], mlp2=[128, 64])
        self.sa5 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=16, in_channel=64 + 3, mlp=[128, 64, 64],
                                          group_all=False)
        self.flow1 = flow_predictor(in_channel=64 + 128, mlp=[128, 64])

        self.drop = nn.Dropout(0.5)
        self.linear1 = nn.Linear(64, 256)

        self.linear2 = nn.Linear(256, 4)
        self.linear3 = nn.Linear(256, 3)

        self.refine1 = RefineBlock_PWCLO(in_channel=64, npoint=256, nsample=4, nsample_q=6, mlp1=[128, 64, 64],
                                         mlp2=[128, 64])
        self.refine2 = RefineBlock_PWCLO(in_channel=32, npoint=1024, nsample=4, nsample_q=6, mlp1=[128, 64, 64],
                                         mlp2=[128, 64])
        self.refine3 = RefineBlock_PWCLO(in_channel=16, npoint=2048, nsample=4, nsample_q=6, mlp1=[128, 64, 64],
                                         mlp2=[128, 64])

    def forward(self, xyz1, xyz2):
        # num_point = xyz1.size(1)# // 2

        l0_xyz_f1 = xyz1[:, :, 0:3]
        # l0_points_f1 = xyz1[:, :, 3:]
        l0_points_f1 = l0_xyz_f1

        l0_xyz_f2 = xyz2[:, :, 0:3]
        # l0_points_f2 = xyz2[:, :, 3:]
        l0_points_f2 = l0_xyz_f2

        # print('-Start-' + ' ' + str(datetime.datetime.now()))
        l1_xyz_f1, l1_points_f1 = self.sa1(l0_xyz_f1, l0_points_f1)
        l2_xyz_f1, l2_points_f1 = self.sa2(l1_xyz_f1, l1_points_f1)
        l3_xyz_f1, l3_points_f1 = self.sa3(l2_xyz_f1, l2_points_f1)
        l4_xyz_f1, l4_points_f1 = self.sa4(l3_xyz_f1, l3_points_f1)

        l1_xyz_f2, l1_points_f2 = self.sa1(l0_xyz_f2, l0_points_f2)
        l2_xyz_f2, l2_points_f2 = self.sa2(l1_xyz_f2, l1_points_f2)
        l3_xyz_f2, l3_points_f2 = self.sa3(l2_xyz_f2, l2_points_f2)
        # l4_xyz_f2, l4_points_f2 = self.sa4(l2_xyz_f2, l2_points_f2)

        l3_feature1_new = self.cost_volume(l3_xyz_f1, l3_xyz_f2, l3_points_f1, l3_points_f2)

        l4_xyz_f1, l4_points_f1_cost_volume = self.sa5(l3_xyz_f1, l3_feature1_new)

        l4_points_predict = l4_points_f1_cost_volume

        l4_cost_volume_w = self.flow1(l4_points_f1, None, l4_points_predict)
        W_l4_feat1 = F.softmax(l4_cost_volume_w, dim=1)

        l4_points_f1_new = torch.sum(l4_points_predict * W_l4_feat1, dim=1, keepdim=False)
        # l4_points_f1_new = l4_points_f1_new.permute(0, 2, 1)
        l4_points_f1_pose = self.linear1(l4_points_f1_new)
        l4_points_f1_q = self.drop(l4_points_f1_pose)
        l4_points_f1_t = self.drop(l4_points_f1_pose)

        l4_q_coarse = self.linear2(l4_points_f1_q)
        l4_q_coarse = l4_q_coarse / (
                    torch.sqrt(torch.sum(l4_q_coarse * l4_q_coarse, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l4_t_coarse = self.linear3(l4_points_f1_t)

        l4_q = torch.squeeze(l4_q_coarse)
        l4_t = torch.squeeze(l4_t_coarse)

        l3_q, l3_t, l3_cost_volume_w, l3_points_predict = self.refine1(l3_xyz_f1, l3_xyz_f2, l4_xyz_f1,
                                                                       l3_points_f1, l3_points_f2, l4_q, l4_t,
                                                                       l4_cost_volume_w, l4_points_predict)
        l2_q, l2_t, l2_cost_volume_w, l2_points_predict = self.refine2(l2_xyz_f1, l2_xyz_f2, l3_xyz_f1,
                                                                       l2_points_f1, l2_points_f2, l3_q, l3_t,
                                                                       l3_cost_volume_w, l3_points_predict)
        l1_q, l1_t, l1_cost_volume_w, l1_points_predict = self.refine3(l1_xyz_f1, l1_xyz_f2, l2_xyz_f1,
                                                                       l1_points_f1, l1_points_f2, l2_q, l2_t,
                                                                       l2_cost_volume_w, l2_points_predict)

        l4_q_norm = l4_q / (torch.sqrt(torch.sum(l4_q * l4_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l3_q_norm = l3_q / (torch.sqrt(torch.sum(l3_q * l3_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l2_q_norm = l2_q / (torch.sqrt(torch.sum(l2_q * l2_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l1_q_norm = l1_q / (torch.sqrt(torch.sum(l1_q * l1_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        return l4_q_norm, l4_t, l3_q_norm, l3_t, l2_q_norm, l2_t, l1_q_norm, l1_t
        # return l4_q_norm, l4_t


if __name__ == '__main__':
    model = get_model()
