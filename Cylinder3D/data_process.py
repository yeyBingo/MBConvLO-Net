# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# from dataloader.pc_dataset import get_SemKITTI_label_name
# from builder import model_builder, loss_builder
# from config.config import load_config_data

# from utils.load_save_util import load_checkpoint
# from dataloader.pc_dataset import SemKITTI_sk
# from dataloader.dataset_semantickitti import collate_fn_BEV_kitti,cylinder_dataset
import warnings
warnings.filterwarnings("ignore")


def ground_segmentation(model, val_grid, val_pt_fea, pytorch_device, config_path='config/semantickitti.yaml'):

    batch_size = len(val_grid)


    config_path = config_path

    # configs = load_config_data(config_path)

    # dataset_config = configs['dataset_params']
    # train_dataloader_config = configs['train_data_loader']
    # val_dataloader_config = configs['val_data_loader']
    #
    # model_config = configs['model_params']
    # train_hypers = configs['train_params']

    # grid_size = model_config['output_shape']
    # num_class = model_config['num_class']
    # ignore_label = dataset_config['ignore_label']


    # data_path = train_dataloader_config["data_path"]
    # val_batch_size = val_dataloader_config['batch_size']
    # val_batch_size = 16

    eval_num = 1
    data_time = 0
    model_time = 0
    label_time = 0
    eval_start = time.time()


    val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                      val_pt_fea]
    val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
    # batch_size = len(val_grid)#.shape[0]

    time1 = time.time()
    predict_labels = model(val_pt_fea_ten, val_grid_ten, batch_size)
    time2 = time.time()
    model_time += time2-time1

    # aux_loss = loss_fun(aux_outputs, point_label_tensor)
    predict_labels = torch.argmax(predict_labels, dim=1)
    predict_labels = predict_labels.cpu().detach().numpy()
    segmented_idx = []


    time3 = time.time()
    label_time += time3 - time2


    del val_grid, val_pt_fea, val_grid_ten
    # save model if performance is improved
    eval_end = time.time()
    eval_time = (eval_end - eval_start) / eval_num
    ave_model_time = model_time/eval_num
    ave_label_time = label_time/eval_num
    # print('Total Time: %.3f(s) Model: %.3f(s) Label: %.3f(s)' %
    #           (eval_time,ave_model_time,ave_label_time))

    # return segmented_idx
    return predict_labels









if __name__ == '__main__':
    # Training settings

    config_path = 'config/semantickitti.yaml'
    ground_segmentation(config_path)
