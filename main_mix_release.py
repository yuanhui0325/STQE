from __future__ import print_function
import argparse
import csv
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import *
from stqe_runtime import STQEModel
from tqdm import tqdm
import datetime
import time
import re
import torch
import torch.nn as nn
#import open3d as o3d
from torch.utils.data import DataLoader
from util import *
from sewar.full_ref import psnr
# from torch.utils.tensorboard import SummaryWriter
devices = "cuda:6"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

            
def train(args, io):
    daytime = datetime.datetime.now().strftime('%Y-%m-%d')  # year,month,day
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    channel = args.train_channel
    yuv_list = ['y', 'u', 'v']
    DATA_DIR = os.path.join(BASE_DIR, args.train_h5_txt)
    DATA_DIR_TEST = os.path.join(BASE_DIR, args.valid_h5_txt)
    logs_path = os.path.join(BASE_DIR,args.log_path + '/STQE/' + daytime + '/' + yuv_list[channel])
    pth_path = os.path.join(BASE_DIR, args.pth_path)
    #writer = SummaryWriter(BASE_DIR + args.tensorboard_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    model_path = pth_path + '/STQE/' + daytime + '/' + yuv_list[channel]
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    txt_loss_mse = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_loss.txt'
    txt_mse_path = os.path.join(logs_path, txt_loss_mse)
    txt_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_loss_rgb.txt'
    txt_psnr_path = os.path.join(logs_path, txt_loss_psnr)
    txt_loss = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_loss.txt'
    txt_loss_path = os.path.join(logs_path, txt_loss)

    # CSV loss logs: initial loss, batch-level loss, and epoch-level loss
    initial_loss_path = os.path.join(logs_path, 'initial_loss.csv')
    train_loss_batch_path = os.path.join(logs_path, 'train_loss_batch.csv')
    train_loss_epoch_path = os.path.join(logs_path, 'train_loss_epoch.csv')

    # initial_loss.csv keeps the loss from the FIRST forward pass of each scratch-training run.
    # The value is recorded before backward() / optimizer.step(), i.e. before any parameter update.
    if not os.path.exists(initial_loss_path):
        with open(initial_loss_path, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                'time', 'epoch', 'iteration', 'lr',
                'initial_mse_loss', 'initial_loss_all', 'initial_mse_loss_ori'
            ])

    # Only create headers when the files do not exist, so resumed training appends safely.
    if not os.path.exists(train_loss_batch_path):
        with open(train_loss_batch_path, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                'epoch', 'iteration', 'global_step', 'lr',
                'mse_loss', 'loss_all', 'mse_loss_ori'
            ])

    if not os.path.exists(train_loss_epoch_path):
        with open(train_loss_epoch_path, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                'epoch', 'lr', 'avg_mse_loss', 'avg_loss_all',
                'avg_mse_loss_ori', 'train_psnr', 'train_psnr_ori'
            ])


    txtValid_name_loss = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_lossValid.txt'
    txt_lossValid_path = os.path.join(logs_path, txtValid_name_loss)
    txt_valid_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_valid_loss_rgb.txt'
    txt_valid_psnr_path = os.path.join(logs_path, txt_valid_loss_psnr)

    traindata, label = load_h5(DATA_DIR)
    dataset = torch.utils.data.TensorDataset(traindata, label)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)

    device = torch.device(devices if args.cuda else "cpu")
    model = STQE().to(device)
    #model.apply(init_weights)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    max_psnr = 0
    if os.path.isfile('%s/model_5.pth' % model_path) :
        print("=> loading checkpoint from epoch 6")

        checkpoint = torch.load('%s/model_5.pth' % model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1  

        max_psnr = checkpoint.get('max_psnr', 0)  
        print("=> loaded checkpoint (epoch {}) with max_psnr: {:.4f}".format(checkpoint['epoch'], max_psnr))

    else:
        print("=> no checkpoint found at epoch 5, starting from scratch")
        start_epoch = 0  

    # Only a true scratch run has an "initial loss".
    # Resumed training (start_epoch > 0) does not overwrite/redefine it.
    initial_loss_recorded_this_run = False
        
    for epoch in range(start_epoch,args.epochs):
    #for epoch in range(args.epochs):

        epoch_loss = AverageMeter()
        epoch_loss_ori = AverageMeter()
        epoch_loss_all = AverageMeter()

        ####################
        # Train
        ####################
        args.lr = args.lr * (0.25 ** (epoch // 60))
        for p in opt.param_groups:
            p['lr'] = args.lr

        model.train()
        with tqdm(total=(traindata.shape[0] - traindata.shape[0] % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, args.epochs))
            for i, (data, label) in enumerate(train_loader, 0):
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.size()[0]

                data = torch.cat((data[:, :, :3, :], torch.unsqueeze(data[:, :, channel + 3, :], dim=2)), dim=2) # batch_size 2048 4 7
                label = label[:, :, channel]

                if len(label.size()) == 2:
                    label = torch.unsqueeze(label, dim=-1)  # batch_size 2048 1
                data = data.permute(0, 2, 1, 3) # batch_size 4 2048 7

                opt.zero_grad()

                rec = data[:, :, :, 0]
                rec = rec.permute(0, 2, 1)[:, :, 3:]  # batch_size 2048 1
                data = torch.autograd.Variable(data, requires_grad=True)

                logits = model(data)
                loss = MSE(logits, label)
                loss_ori = MSE(rec, label)
                loss_all = criterion(logits, label, alpha=1)

                # Record the very first loss BEFORE any backward/update.
                # This is the actual starting loss of a scratch-training run.
                if start_epoch == 0 and epoch == 0 and i == 0 and not initial_loss_recorded_this_run:
                    current_lr = opt.param_groups[0]['lr']
                    with open(initial_loss_path, 'a', newline='') as f:
                        writer_csv = csv.writer(f)
                        writer_csv.writerow([
                            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                            epoch, i, current_lr,
                            loss.item(), loss_all.item(), loss_ori.item()
                        ])
                    print(
                        'Initial Loss (before first update) | '
                        'MSEloss:{:.10f} loss_all:{:.10f} MSEloss_ori:{:.10f}'.format(
                            loss.item(), loss_all.item(), loss_ori.item()
                        )
                    )
                    initial_loss_recorded_this_run = True

                with torch.autograd.set_detect_anomaly(True):
                    loss_all.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                opt.step()
                epoch_loss.update(loss.item(), len(rec))
                epoch_loss_ori.update(loss_ori.item(), len(rec))
                epoch_loss_all.update(loss_all.item(), len(rec))

                # Record every training iteration for plotting detailed loss curves.
                global_step = epoch * len(train_loader) + i
                current_lr = opt.param_groups[0]['lr']
                with open(train_loss_batch_path, 'a', newline='') as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([
                        epoch, i, global_step, current_lr,
                        loss.item(), loss_all.item(), loss_ori.item()
                    ])

                _tqdm.set_postfix(loss='{:.7f}'.format(epoch_loss.avg))
                _tqdm.update(len(data))

        scheduler.step()

        epoch_psnr = mse2psnr(epoch_loss.avg)
        epoch_psnr_ori = mse2psnr(epoch_loss_ori.avg)

        # Record one averaged training-loss row per epoch.
        current_lr = opt.param_groups[0]['lr']
        with open(train_loss_epoch_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                epoch, current_lr, epoch_loss.avg, epoch_loss_all.avg,
                epoch_loss_ori.avg, epoch_psnr, epoch_psnr_ori
            ])

        file_log = open(txt_mse_path, 'a')
        print('epoch:{} loss_all:{} MSEloss:{} MSEloss_ori:{}'.format(epoch, epoch_loss_all.avg, epoch_loss.avg, epoch_loss_ori.avg), file=file_log)
        file_log.close()

        print('epoch:{}'.format(epoch), 'average MSEloss_all:{}'.format(epoch_loss_all.avg),'MSEloss:{}'.format(epoch_loss.avg),'MSEloss_ori:{}'.format(epoch_loss_ori.avg))

        file_loss = open(txt_psnr_path, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_psnr), 'psnr_origin:{}'.format(epoch_psnr_ori),
              file=file_loss)
        file_loss.close()


        if epoch >= 5 and epoch_psnr - epoch_psnr_ori > max_psnr:  # save the model with max PSNR promotion
            max_psnr = epoch_psnr - epoch_psnr_ori
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss,
                "max_psnr": max_psnr,  # Save max_psnr
            },
                '%s/model_%d.pth' % (model_path, epoch)
            )




def test(args, io):
    device = torch.device(devices)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR_TEST = os.path.join(BASE_DIR, args.test_ply_txt)
    ori_path = os.path.join(BASE_DIR, args.test_ori_ply)
    rec_base = os.path.join(BASE_DIR, args.test_rec_ply)
    h5_dir = os.path.join(BASE_DIR, args.test_h5)

    model1 = STQEModel(
        args.model1_path,
        device=device
    )

    model2 = STQEModel(
        args.model2_path,
        device=device
    )

    model3 = STQEModel(
        args.model3_path,
        device=device
    )

    textfile_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_test_STQE.txt'
    LOG_FOUT = open(os.path.join(args.log_path_test, textfile_name), 'w')
    LOG_FOUT.write(str(args) + '\n')

    p_num = 2048 
    iter = 0
    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)

    with open(DATA_DIR_TEST, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            log_string(LOG_FOUT, line)
            iter += 1
            log_string(LOG_FOUT, 'sequence: %s, iter: %d' % (line, iter))

            if os.path.splitext(line)[1] == ".h5":
                ply_name = re.sub(r'_r\d\d', '', line)  
                ply_name = os.path.splitext(ply_name)[0] + ".ply"
                pointcloud_ori = read_ply(os.path.join(ori_path, ply_name))
                ori_color = rgb2yuv(pointcloud_ori[:, 3:]).astype(np.float32)
                ori_loc = pointcloud_ori[:, :3].astype(np.float32)
                pointNum = ori_loc.shape[0]
                
                m = re.match(r"^(.*)_r(\d\d)_(\d+)\.h5$", line)
                if m:
                    base_name = m.group(1)      
                    rate = m.group(2)           
                    frame_str = m.group(3)     
                    rec_frame = str(int(frame_str)).zfill(4) 
                    rec_ply_name = "reconstructed-" + rec_frame + ".ply"
                    rec_ply_path = os.path.join(rec_base, base_name, "r" + rate, rec_ply_name)
                else:
                    log_string(LOG_FOUT, "Filename pattern not match: " + line)
                    continue

                if not os.path.exists(rec_ply_path):
                    log_string(LOG_FOUT, "Reconstructed ply file not found: " + rec_ply_path)
                    continue

                rec_ply = read_ply(rec_ply_path)
                rec_loc = rec_ply[:, :3].astype(np.float32) 
                rec_color = np.array(rgb2yuv(rec_ply[:, 3:])).astype(np.float32)

                h5_file_path = os.path.join(h5_dir, line)
                with h5py.File(h5_file_path, 'r') as hf:
                    data = hf['data'][:]   
                    group_idx = hf['group_idx'][:] 
                numPatch = data.shape[0]
                batch_size = args.test_batch_size

                new_color1 = torch.zeros(pointNum, dtype=torch.float32).to(device)
                new_color2 = torch.zeros(pointNum, dtype=torch.float32).to(device)
                new_color3 = torch.zeros(pointNum, dtype=torch.float32).to(device)
                count = torch.zeros(pointNum, dtype=torch.float32).to(device)

                for i in range(0, numPatch, batch_size):
                    end_i = min(i + batch_size, numPatch)
                    current_batch_size = end_i - i

                    batch_data = data[i:end_i, :, :, :]
                    batch_group_idx = group_idx[i:end_i, :]  # shape: (B, 2048)


                    input1 = np.concatenate((batch_data[:, :, :3, :], batch_data[:, :, 3:4, :]), axis=2)  # (B, 2048, 4, 3)
                    input2 = np.concatenate((batch_data[:, :, :3, :], batch_data[:, :, 4:5, :]), axis=2)
                    input3 = np.concatenate((batch_data[:, :, :3, :], batch_data[:, :, 5:6, :]), axis=2)

                    input1 = np.transpose(input1, (0, 2, 1, 3))
                    input2 = np.transpose(input2, (0, 2, 1, 3))
                    input3 = np.transpose(input3, (0, 2, 1, 3))

                    input1_tensor = torch.tensor(input1, dtype=torch.float32).to(device)
                    input2_tensor = torch.tensor(input2, dtype=torch.float32).to(device)
                    input3_tensor = torch.tensor(input3, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        preds1 = model1(input1_tensor)
                        preds2 = model2(input2_tensor)
                        preds3 = model3(input3_tensor)
                    preds1 = torch.squeeze(preds1, dim=-1)  # (B, 2048)
                    preds2 = torch.squeeze(preds2, dim=-1)
                    preds3 = torch.squeeze(preds3, dim=-1)
                    
                    for b in range(current_batch_size):
                        patch_indices = batch_group_idx[b]  
                        new_color1[patch_indices] += preds1[b]
                        new_color2[patch_indices] += preds2[b]
                        new_color3[patch_indices] += preds3[b]
                        count[patch_indices] += 1

                mask = count > 0
                print("Unpredicted Points Num: ", torch.sum(~mask).item())
                new_color1[mask] = new_color1[mask] / count[mask]
                new_color2[mask] = new_color2[mask] / count[mask]
                new_color3[mask] = new_color3[mask] / count[mask]
                rec_color_tensor = torch.tensor(rec_color, dtype=torch.float32).to(device)
                new_color1[~mask] = rec_color_tensor[~mask, 0]
                new_color2[~mask] = rec_color_tensor[~mask, 1]
                new_color3[~mask] = rec_color_tensor[~mask, 2]

                pred_yuv = torch.stack([new_color1, new_color2, new_color3], dim=1)
                pred_yuv_np = pred_yuv.cpu().numpy()
                output_color = np.clip(np.round(yuv2rgb(pred_yuv_np)), 0, 255)

                psnr_ori1 = psnr(ori_color[:, 0], rec_color[:, 0], MAX=255)
                psnr_pred1 = psnr(ori_color[:, 0], rgb2yuv(output_color)[:, 0], MAX=255)
                psnr_ori2 = psnr(ori_color[:, 1], rec_color[:, 1], MAX=255)
                psnr_pred2 = psnr(ori_color[:, 1], rgb2yuv(output_color)[:, 1], MAX=255)
                psnr_ori3 = psnr(ori_color[:, 2], rec_color[:, 2], MAX=255)
                psnr_pred3 = psnr(ori_color[:, 2], rgb2yuv(output_color)[:, 2], MAX=255)
                log_string(LOG_FOUT, "psnr_y for original:  %f" % psnr_ori1)
                log_string(LOG_FOUT, "psnr_y for pred:  %f" % psnr_pred1)
                log_string(LOG_FOUT, "psnr_u for original:  %f" % psnr_ori2)
                log_string(LOG_FOUT, "psnr_u for pred:  %f" % psnr_pred2)
                log_string(LOG_FOUT, "psnr_v for original:  %f" % psnr_ori3)
                log_string(LOG_FOUT, "psnr_v for pred:  %f" % psnr_pred3)

                output = rec_ply.copy()
                output[:, 3:] = output_color
                filepath = os.path.join(args.pred_path, ply_name)
                write_ply(output, filepath)

                log_string(LOG_FOUT, "Processed point cloud: %s" % ply_name)

    LOG_FOUT.close()



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud quality Enhancement')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--pth_path', type=str, default='/data/pths')
    parser.add_argument('--log_path', type=str, default='/data/logs')
    #parser.add_argument('--tensorboard_path', type=str, default='/tensorboard_log_s80v')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Learning rate')
    parser.add_argument('--log_path_test', type=str, default='/data/raht/logs_test')
    parser.add_argument('--train_h5_txt', type=str, default='/disk/DATA/h5RAHT/Train/fastr3.txt')
    parser.add_argument('--test_ply_txt', type=str, default='/disk/DATA/h5RAHT/Test/testFiler6.txt')
    parser.add_argument('--test_ori_ply', type=str, default='/disk/DATA/Ori_sameOrder/')
    parser.add_argument('--test_h5', type=str, default='/disk/DATA/h5RAHT/Test/')
    parser.add_argument('--test_rec_ply', type=str, default='/disk/DATA/recRAHT/')
    parser.add_argument('--train_channel', type=int, default=0, help='0:Y, 1:U, 2:V')
    parser.add_argument('--valid_h5_txt', type=str, default='/disk/DATA/h5RAHT/Train/uvvalidFiler1.txt')
    parser.add_argument('--model', type=str, default='STQE', metavar='N',
                        help='Model to use, STQE')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='test_batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--has_model', type=bool, default=False,
                        help='Checkpoints')

    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluate the model (Test stage)')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model1_path', type=str, default='./pretrained/stqe_y_r04.stqe',
                        metavar='N',
                        help='Pretrained model1 path')
    parser.add_argument('--model2_path', type=str, default='./pretrained/stqe_u_r04.stqe',
                        metavar='N',
                        help='Pretrained model2 path')
    parser.add_argument('--model3_path', type=str, default='./pretrained/stqe_v_r04.stqe',
                        metavar='N',
                        help='Pretrained model3 path')
    parser.add_argument('--pred_path', type=str, default='/data/raht/predData/r6')
    
    parser.add_argument("--K1", type=int, default=20)
    args = parser.parse_args()
    
    set_seed(args.seed)

    _init_()
    print(args.eval)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device(devices)
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')

        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)