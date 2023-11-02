"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from tqdm import tqdm
import sys

from torch.autograd import Variable
import MBConvLO_Net
import kitti_dataset
import datetime
from evaluation import kittieval


parser = argparse.ArgumentParser(description='MBConvLO-Net Training and Testing')
parser.add_argument('--data_root', default='../dataset/kitti_odometry', help='Path to dataset directory ')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

parser.add_argument('--mode', default='test', help='train/test mode')
parser.add_argument('--ground_removal', default=False, type=bool, help='use ground removal or not')
parser.add_argument('--gpu', default=5, type=int, help='GPU id to use.')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--ckpt', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 8]')
parser.add_argument('--lr', '--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--lr_momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Decay rate for lr decay [default: 1e-6]')

parser.add_argument('--train_list', nargs='+', type=int, default=range(7), help=' List of sequences for training [default: range(7)]')
parser.add_argument('--val_list', nargs='+',  type=int, default=range(7,11), help=' List of sequences for validation [default: range(7, 11)]')
parser.add_argument('--test_list', nargs='+', type=int, default=range(11), help='List of sequences for testing [default: range(11)]')

#evaluation
parser.add_argument('--gt_dir', type=str, default='./ground_truth_pose', help='Directory path of the ground truth odometry')
parser.add_argument('--result_dir', type=str, default='./result', help='Directory path of storing the odometry results')
parser.add_argument('--toCameraCoord', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Whether to convert the pose to camera coordinate')

args = parser.parse_args()
device = torch.device("cuda:"+str(args.gpu))

LOG_DIR = 'logfiles/log' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')

NUM_POINT = args.num_point
DATA = args.data_root


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def log_only(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.
    References
    '''
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return torch.eye(3)
    s = 2.0 / Nq
    X,Y,Z = x * s, y * s, z * s
    wX,wY,wZ = w * X, w * Y, w * Z
    xX,xY,xZ = x * X, x * Y, x * Z
    yY,yZ,zZ = y * Y, y * Z, z * Z
    return torch.Tensor(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])

def main():

    TRAIN_LIST = args.train_list
    VAL_LIST = args.val_list
    TEST_LIST = args.test_list
    MODE = args.mode
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    train_dataset = kitti_dataset.OdometryDataset(DATA, npoints=NUM_POINT, is_training=True, sequences=TRAIN_LIST)
    val_dataset = kitti_dataset.OdometryDataset(DATA, npoints=NUM_POINT, is_training=False, sequences=VAL_LIST)
    test_dataset = kitti_dataset.OdometryDataset(DATA, npoints=NUM_POINT, is_training=False, sequences=TEST_LIST)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        print("Use Seed: {} for training".format(args.seed))
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    # create model
    model = MBConvLO_Net.get_model(args)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # define loss function (criterion) and optimizer
    w_x = Variable(torch.Tensor([0.0]), requires_grad=True).to(device)
    w_q = Variable(torch.Tensor([-2.5]), requires_grad=True).to(device)
    criterion = MBConvLO_Net.get_loss3(w_x, w_q)


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    model = model.to(device)

    optimizer = None
    if args.optimizer == 'momentum':
        optimizer = torch.optim.SGD([{'params':model.parameters()},
                                      {'params':criterion.parameters()}],
                                        lr=args.lr, momentum=args.lr_momentum, weight_decay=args.weight_decay)

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params':model.parameters()},
                                      {'params': criterion.parameters()}],
                                        lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, threshold=100,
                                               threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-08, verbose=True)


    if args.ckpt:
        args.ckpt = 'checkpoint/' + args.ckpt
        if os.path.isfile(args.ckpt):
            print("=> loading checkpoint '{}'".format(args.ckpt))
            checkpoint = torch.load(args.ckpt)
            args.start_epoch = checkpoint['epoch']
            best_error = checkpoint['best_error']

            model.load_state_dict(checkpoint['model_state_dict'])
            criterion.load_state_dict(checkpoint['criterion_state_dict'])

        else:
            print("=> no checkpoint found at '{}'".format(args.ckpt))

    cudnn.benchmark = True



    train_sampler = None
    val_sampler = None
    test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False)
    log_string('[main Model] '+ model.name)


    if MODE == 'train':
        for epoch in range(0, args.epochs):#args.start_epoch
            # train for one epoch

            train(train_loader, model, criterion, optimizer, scheduler, epoch)

            if (epoch) % 5 == 0:
                cur_eval_error = validate(val_loader, model, criterion, epoch, args)

            # if cur_eval_error < min_eval_error:
                min_eval_error = cur_eval_error
                save_dict = {'epoch': epoch,
                             'model_state_dict': model.state_dict(),
                             'criterion_state_dict': criterion.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'best_error': cur_eval_error,

                             }
                dict_name =  'Main_'+ model.name + "_" + str(cur_eval_error) + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') + ".pth"
                torch.save(save_dict, "./checkpoint/" + dict_name)
                log_string('***{} Saved***'.format(dict_name))


    elif MODE == 'test':
        validate(test_loader, model, criterion, None, args)


def train(data_loader, model, criterion, optimizer, scheduler, epoch):

    log_string('[epoch '+ str(epoch) + ']' +'Learning rate:' + str(optimizer.param_groups[0]['lr']))

    model.train()  # 更新参数
    batch_loss = 0.0
    best_loss = 10000

    data_loader = tqdm(data_loader, file=sys.stdout)


    for i, data in enumerate(data_loader, 0):

        pos1, pos2, q_gt, t_gt = data
        pos1 = pos1.to(device).float()
        pos2 = pos2.to(device).float()
        q_gt = q_gt.to(device)
        t_gt = t_gt.to(device)

        outputs = model(pos1, pos2)
        odometry_loss = criterion(outputs, q_gt, t_gt)
        loss = odometry_loss

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()

        data_loader.desc = "[ {} ]loss:{:.3f}".format(epoch, loss.item())
        if (i + 1) % 10 == 0 :
            mean_loss = batch_loss / 10
            log_string('[ mean loss:{:.3f} ]'.format(mean_loss))
            if mean_loss < best_loss:
                best_loss = mean_loss

            batch_loss = 0.0


    log_string('[Epoch {}] best loss: {:.4f}'.format(epoch, best_loss))

    scheduler.step(best_loss)



@torch.no_grad()
def validate(val_loader, model, criterion, epoch, args):
    model.eval()

    running_loss = 0.0

    s = [0, 4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000]
    e = [4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000, 23201]
    val_num = 0
    batch_num = 0

    with torch.no_grad():


            start_time = time.time()
            start = s[ii]
            end = e[ii]

            if args.ground_removal == True:

                from Cylinder3D.config.config import load_config_data
                from Cylinder3D.dataloader.pc_dataset import SemKITTI_sk
                from Cylinder3D.dataloader.dataset_semantickitti import collate_fn_BEV_kitti, cylinder_dataset
                from Cylinder3D.builder import model_builder, loss_builder
                from Cylinder3D.data_process import ground_segmentation
                config_path = './Cylinder3D/config/semantickitti.yaml'
                configs = load_config_data(config_path)
                dataset_config = configs['dataset_params']
                model_config = configs['model_params']
                grid_size = model_config['output_shape']
                model_config = configs['model_params']
                train_hypers = configs['train_params']
                model_load_path = train_hypers['model_load_path']
                seg_model = model_builder.build(model_config)
                if os.path.exists(model_load_path):
                    seg_model.load_state_dict(torch.load(model_load_path))
                    # seg_model = load_checkpoint(model_load_path, seg_model)
                    print("=> loading checkpoint '{}'".format(model_load_path))
                pytorch_device = torch.device('cuda:' + str(args.gpu))

                seg_model.to(pytorch_device)
                seg_model.eval()

                # Loading pointcloud
                pt_dataset = SemKITTI_sk(data_path=DATA, imageset='val', return_ref=True,
                                             label_mapping='./Cylinder3D/config/label_mapping/semantic-kitti.yaml', sequence=ii)
                # Point voxelization
                voxel_dataset = cylinder_dataset(
                    pt_dataset,
                    grid_size=grid_size,
                    fixed_volume_space=dataset_config['fixed_volume_space'],
                    max_volume_space=dataset_config['max_volume_space'],
                    min_volume_space=dataset_config['min_volume_space'],
                    ignore_label=dataset_config["ignore_label"],
                )

                data_loader = torch.utils.data.DataLoader(dataset=voxel_dataset,
                                                                 batch_size=args.batch_size,
                                                                 collate_fn=collate_fn_BEV_kitti,
                                                                 shuffle=False,
                                                                 num_workers=4)
            else:
                val_dataset = kitti_dataset.OdometryDataset(root_path=DATA, npoints=NUM_POINT, is_training=False, sequences=range(ii,ii+1))
                data_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=False)

            log_string('==== LIST %03d EVALUATION ====' % (ii))
            tmp = 0

            for i, data in enumerate(data_loader, 0):


                if args.ground_removal == True:
                    grid_ind, return_fea, ori_pos1, ori_pos2, q_gt, t_gt = data
                    predict_labels = ground_segmentation(seg_model, grid_ind, return_fea, pytorch_device,
                                                 config_path=config_path)

                    for count, _ in enumerate(grid_ind):
                        panoptic = predict_labels[
                            count, grid_ind[count][:, 0], grid_ind[count][:, 1], grid_ind[count][:, 2]]
                        not_ground = (panoptic == 1)
                        segmented_idx = np.where(not_ground)[0]

                        pos_tmp = ori_pos1[count][segmented_idx, :]
                        sample_idx = np.linspace(0, pos_tmp.shape[0] - 1, NUM_POINT, dtype=int)
                        pos_tmp = np.expand_dims(pos_tmp[sample_idx, :],axis=0)

                        if count == 0:
                            pos1 = pos_tmp
                        else:
                            pos1 = np.concatenate([pos1, pos_tmp], axis=0)

                    if i == 0:
                        pos2 = np.concatenate([pos1[0:1,:,:], pos1[:-1,:,:]], axis=0)
                    else:
                        pos2 = np.concatenate([last_pos1, pos1[:-1,:,:]], axis=0)

                    last_pos1 = pos1[-1:,:,:]
                    pos1 = torch.from_numpy(pos1).to(device).float()
                    pos2 = torch.from_numpy(pos2).to(device).float()
                else:
                    pos1, pos2, q_gt, t_gt = data
                    pos1 = pos1.to(device).float()
                    pos2 = pos2.to(device).float()
                if i % 50 == 0:
                    log_string(str(datetime.datetime.now()))
                    log_string('-batch %03d- in evaluation'%(i))


                q_gt = q_gt.to(device)
                t_gt = t_gt.to(device)

                # model_start = time.time()
                outputs = model(pos1, pos2)
                # print('model time:',time.time()-model_start)
                loss = criterion(outputs, q_gt, t_gt)


                running_loss += loss.item()

                pred_q, pred_t = outputs[-2:]

                if len(pred_q.shape) == 1:
                    pred_q = pred_q.unsqueeze(0)
                    pred_t = pred_t.unsqueeze(0)

                for n in range(0,pred_q.shape[0]):

                    if pred_q.shape[0] != 1:
                        q_one_batch = pred_q[n:n + 1, :]
                        t_one_batch = pred_t[n:n + 1, :]
                    else:
                        q_one_batch = pred_q
                        t_one_batch = pred_t

                    qq = torch.reshape(q_one_batch, (4,))
                    tt = torch.reshape(t_one_batch, (3, 1))

                    RR = quat2mat(qq).to(device)

                    filler = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(device)
                    filler = torch.unsqueeze(filler, dim=0)  ##1*4

                    TT = torch.cat([torch.cat([RR, tt], dim=-1), filler], dim=0)

                    if tmp == 0:

                        T_final = TT  ### 4 4
                        T = T_final[:3, :]  ####  3 4
                        T = T.reshape(1, 1, 12)
                        tmp += 1

                    else:
                        T_final = torch.matmul(T_final, TT)
                        T_current = T_final[:3, :]
                        T_current = T_current.reshape(1, 1, 12)
                        T = torch.cat((T, T_current), dim=0)

            batch_num += i
            val_num += end - start

            T = T.reshape(-1, 12)

            fname_txt = os.path.join(LOG_DIR, str(ii).zfill(2) + '_pred.txt')
            result_dir = args.result_dir

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            T = T.cpu()
            np.savetxt(fname_txt, T, fmt='%.08f')
            os.system('cp %s %s' % (fname_txt, result_dir))  ###  SAVE THE txt FILE

            ave_t_err, ave_r_err = kittieval(ii,args)

            log_string("seq" + str(ii).zfill(2) + " Average_t_error {0:.2f}".format(
                ave_t_err * 100) + " Average_r_error {0:.2f}".format(ave_r_err / np.pi * 180 * 100))

            end_time = time.time()
            loss_mean = running_loss / (val_num)
            time_mean = (end_time - start_time) / val_num
            fps_mean = 1 / time_mean
            log_string('[Running time] time_mean: %.2f(ms) FPS: %.2f' % (time_mean * 1000, fps_mean))
            val_num = 0


    log_string('[valid epoch %d] loss_mean: %.5f' %(epoch,loss_mean ))

    return loss_mean





if __name__ == '__main__':
    main()

