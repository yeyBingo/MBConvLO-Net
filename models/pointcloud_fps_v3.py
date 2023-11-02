# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import random
# import time
# from numba import cuda
# import torch
#
# def readXYZfile(filename, Separator):
#     data = [[], [], []]
#     f = open(filename, 'r')
#     line = f.readline()
#     num = 0
#     while line:  # 按行读入点云
#         c, d, e, _, _, _ = line.split(Separator)
#         data[0].append(c)  # x坐标
#         data[1].append(d)  # y坐标
#         data[2].append(e)  # z坐标
#         num = num + 1
#         line = f.readline()
#     f.close()
#
#     #string型转float型
#     x = [float(data[0]) for data[0] in data[0]]
#     z = [float(data[1]) for data[1] in data[1]]
#     y = [float(data[2]) for data[2] in data[2]]
#     print("读入点的个数为：{}个。".format(num))
#     point = [x, y, z]
#     return point
#
# # 三维离散点图显示点云
# def displayPoint(data, title):
#     # 解决中文显示问题
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     # 点数量太多不予显示
#     while len(data[0]) > 20000:
#         print("点太多了！")
#         exit()
#     #散点图参数设置
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.set_title(title)
#     ax.scatter3D(data[0], data[1], data[2], c = 'b', marker = '.')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()
#
# def displayPoint2(data,sample_data, title):
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     while len(data[0]) > 20000:
#         print("点太多了！")
#         exit()
#
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.set_title(title)
#     ax.scatter3D(data[0], data[1], data[2], c = 'b', marker = '.')
#     ax.scatter3D(sample_data[0], sample_data[1], sample_data[2], c = 'r', marker = '.')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()
#
# # step1
# @cuda.jit
# def getFarthestPointKernel(d_data, d_temp, ind, d_result, d_farthest_point, n):
#     idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
#     stride = cuda.gridDim.x * cuda.blockDim.x
#     i = idx
#
#     if i < n:
#         if i != ind:
#             length = (d_data[ind][0] - d_data[i][0]) ** 2 + (d_data[ind][1] - d_data[i][1]) ** 2 + (d_data[ind][2] - d_data[i][2]) ** 2
#             d_temp[i] = length
#         cuda.syncthreads()
#         cuda.atomic.max(d_result, 0, d_temp[i])
#
#         if d_temp[i] == d_result[0]:
#             d_farthest_point[0] = i
#             d_temp[i] = -1.0
# # step2
# @cuda.jit
# def getFarthestPointKernel2(d_data, d_temp, d_rest, d_select, d_step2_result, d_farthest_point, rest_lens, select_lens):
#     idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
#     stride = cuda.gridDim.x * cuda.blockDim.x
#     i = idx
#     j = select_lens - 1
#
#     if i < rest_lens:
#         length = (d_data[d_rest[i]][0] - d_data[d_select[j]][0]) ** 2 + (d_data[d_rest[i]][1] - d_data[d_select[j]][1]) ** 2 + (d_data[d_rest[i]][2] - d_data[d_select[j]][2]) ** 2
#         cuda.syncthreads()
#
#         if length < d_temp[d_rest[i]]:
#             d_temp[d_rest[i]] = length
#         cuda.syncthreads()
#         cuda.atomic.max(d_step2_result, 0, d_temp[d_rest[i]])
#
#         if d_temp[d_rest[i]] == d_step2_result[0]:
#             d_farthest_point[0] = d_rest[i]
#             d_temp[d_rest[i]] = -1.0
#
# # FarthestPointSampling
# def farthestPointSampling(points, samples_num):
#
#     # points = np.ascontiguousarray(points.T)
#     select = []
#     lens = points.shape[0]
#     rest = [num for num in range(0, lens)]
#     max_dist = -1e10
#     farthest_point = np.ones(1, dtype=np.int64)
#     temp = np.zeros(lens, dtype=np.float64)
#     result = np.zeros(1, dtype=np.float64)
#
#     random.seed(1)
#     ind = random.randint(0, lens)
#     select.append(ind)
#     rest.remove(ind)
#
#     threadsperblock = 256
#     blockspergrid = (lens + (threadsperblock - 1)) // threadsperblock
#
#     d_data = cuda.as_cuda_array(points)
#     # d_temp = cuda.as_cuda_array(temp)
#     # d_result = cuda.as_cuda_array(result)
#     # d_farthest_point = cuda.as_cuda_array(farthest_point)
#     # d_data = cuda.to_device(points)
#     d_temp = cuda.to_device(temp)
#     d_result = cuda.to_device(result)
#     d_farthest_point = cuda.to_device(farthest_point)
#
#     getFarthestPointKernel[threadsperblock, blockspergrid](d_data, d_temp, ind, d_result, d_farthest_point, lens)
#     cuda.synchronize()
#     farthest_point = d_farthest_point.copy_to_host()
#
#     select.append(farthest_point[0])
#     rest.remove(farthest_point[0])
#
#     while len(select) <  samples_num:
#
#         rest_lens = len(rest)
#         select_lens = len(select)
#         step2_result = np.zeros(1, dtype=np.float64)
#         threadsperblock = 256
#         blockspergrid2 = (rest_lens + (threadsperblock - 1)) // threadsperblock
#         d_rest = cuda.to_device(rest)
#         d_select = cuda.to_device(select)
#         d_step2_result = cuda.to_device(step2_result)
#
#         getFarthestPointKernel2[threadsperblock, blockspergrid2](d_data, d_temp, d_rest, d_select, d_step2_result, d_farthest_point, rest_lens, select_lens)
#         cuda.synchronize()
#         farthest_point = d_farthest_point.copy_to_host()
#
#         select.append(farthest_point[0])
#         rest.remove(farthest_point[0])
#
#     # new_x = []
#     # new_y = []
#     # new_z = []
#     # for i in range(len(select)):
#     #     new_x.append(points[0][select[i]])
#     #     new_y.append(points[1][select[i]])
#     #     new_z.append(points[2][select[i]])
#
#     # point = [new_x, new_y, new_z]
#     point = points[select,:]
#     return point
#
#
# if __name__ == "__main__":
#
#     point1 = np.fromfile('../pos1.bin', dtype=np.float32).reshape(-1, 4)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     xyz = torch.from_numpy(point1[:,:3]).to(device)
#     # xyz = point1[:,:3]
#     # xyz = readXYZfile("airplane_0032.txt", ',')
#     #displayPoint(data, "airplane")
#     print("V3 版本")
#     print("Sample点数 1000")
#     start_gpu = time.time()
#     print(start_gpu)
#     sample_data = farthestPointSampling(xyz, 256)
#     end_gpu = time.time()
#     print(end_gpu)
#
#     print('------ Gpu process time:' + str(end_gpu - start_gpu))
#
#     # displayPoint(sample_data, "airplane")

# import numpy as np
# import time
# import torch
# from torch.autograd import Variable

# def farthest_point_sample(xyz, npoint):
#
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#
#     # xyz = xyz.transpose(2,1)
#     device = xyz.device
#     B, N, C = xyz.shape
#
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)     # 采样点矩阵（B, npoint）
#     distance = torch.ones(B, N).to(device) * 1e10                       # 采样点到所有点距离（B, N）
#
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)        # batch_size 数组
#
#     #farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 初始时随机选择一点
#
#     barycenter = torch.sum((xyz), 1)                                    #计算重心坐标 及 距离重心最远的点
#     barycenter = barycenter/xyz.shape[1]
#     barycenter = barycenter.view(B, 1, 3)
#
#     dist = torch.sum((xyz - barycenter) ** 2, -1)
#     farthest = torch.max(dist,1)[1]                                     #将距离重心最远的点作为第一个点
#
#     for i in range(npoint):
#         # print("-------------------------------------------------------")
#         # print("The %d farthest pts %s " % (i, farthest))
#         centroids[:, i] = farthest                                      # 更新第i个最远点
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)        # 取出这个最远点的xyz坐标
#         dist = torch.sum((xyz - centroid) ** 2, -1)                     # 计算点集中的所有点到这个最远点的欧式距离
#         # print("dist    : ", dist)
#         mask = dist < distance
#         # print("mask %i : %s" % (i,mask))
#         distance[mask] = dist[mask]                                     # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
#         # print("distance: ", distance)
#
#         farthest = torch.max(distance, -1)[1]                           # 返回最远点索引
#
#     return centroids

import numpy as np
import time
import torch
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from pointnet2_util import index_points
import datetime
if __name__ == '__main__':
    point1 = np.fromfile('../pos1.bin', dtype=np.float32).reshape(-1, 4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xyz = torch.from_numpy(point1[:,:3]).to(device)
    xyz = xyz.unsqueeze(0)
    # sim_data = Variable(torch.rand(1,3,8))
    # print(xyz)
    for i in range(10):

        # start_gpu = time.time()
        start_gpu = datetime.datetime.now()
        print(start_gpu)
        idx = furthest_point_sample(xyz, 2048)
        idx = idx.long()
        new_xyz = index_points(xyz,idx)

        # centroids = farthest_point_sample(xyz, 256)
        # end_gpu = time.time()
        end_gpu = datetime.datetime.now()
        print(end_gpu)
        print('------ Gpu process time:' + str(end_gpu - start_gpu))
    xyz[0].cpu().detach().numpy().astype('float32').tofile('./fps_pos1.bin')
    new_xyz[0].cpu().detach().numpy().astype('float32').tofile('./fps_pos1_sim.bin')


    # print("Sampled pts: ", centroids)



