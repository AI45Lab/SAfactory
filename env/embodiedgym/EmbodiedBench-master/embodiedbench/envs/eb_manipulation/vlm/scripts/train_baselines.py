import datetime
from email.policy import strict
import os
from pickle import NONE
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch._C import device
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import argparse
import warnings
from distutils.util import strtobool

import sys
from os.path import join, dirname, abspath, isfile

CURRENT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(CURRENT_DIR, '..'))  # Use local amsolver rather than installed

from cliport.agent import BlindLangAgent_6Dof, ImgDepthAgent_6dof, TwoStreamClipLingUNetLatTransporterAgent
warnings.filterwarnings('ignore')
import torch.nn.functional as F

from vlm.scripts.VLDataloader import VLM_dataset
from vlm.scripts.eval_sampler import DistributedEvalSampler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def collate_fn(batch):
    return batch

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename+'.pth')
    if is_best:
        torch.save(state, 'model_best.pth')

def sec_to_str(delta):
    t = datetime.timedelta(seconds=delta)
    s = str(t)
    return s.split(".")[0] + "s"

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def main(args):

    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # if args.workers >= 1:
    #     try: mp.set_start_method('spawn')
    #     except RuntimeError: print("multiprocessing method is already set.")

    ngpus_per_node = torch.cuda.device_count() if args.gpu_number==0 else args.gpu_number
    args.ngpus_per_node = ngpus_per_node
    # ngpus_per_node = 5
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu + args.gpu_start
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        # os.environ["MASTER_ADDR"] = "127.0.0.1"
        # os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    loss_func = None
    cfg = {
            'train':{
                'attn_stream_fusion_type': 'add',
                'trans_stream_fusion_type': 'conv',
                'lang_fusion_type': 'mult',
                'n_rotations':36,
                'batchnorm':False
            }
        }
    device = torch.device(args.gpu)
    if args.baseline_mode == 'cliport_6dof':
        model = TwoStreamClipLingUNetLatTransporterAgent(name="cliport_6dof",device=device, cfg=cfg, z_roll_pitch=True)
    elif args.baseline_mode == 'imgdepth_6dof':
        model = ImgDepthAgent_6dof(name=args.baseline_mode,device=device, cfg=cfg)
    elif args.baseline_mode == 'blindlang_6dof':
        model = BlindLangAgent_6Dof(name=args.baseline_mode,device=device, cfg=cfg)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model.cuda(args.gpu)
    parameters = [p for name, p in model.named_parameters() if p.requires_grad]
    # parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            if args.distributed:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            # try:
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            # except:
            #     pass
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True
    train_dataset = VLM_dataset(args.data_dir, 'train', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
                    use_fail_cases = args.use_fail_cases, sample_numbers = args.sample_numbers, train_tasks=args.train_tasks, args=args)
    val_dataset = VLM_dataset(args.data_dir, 'valid', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
                    use_fail_cases = args.use_fail_cases, sample_numbers = args.sample_numbers, train_tasks=args.train_tasks, args=args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = DistributedEvalSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=args.pin_memory, sampler=train_sampler, 
        drop_last=True, collate_fn = collate_fn, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=args.pin_memory, sampler=val_sampler, 
        drop_last=True, collate_fn = collate_fn, persistent_workers=True)

    if args.wandb_entity is not None and args.rank==0:
        import wandb
        run_name = f'{args.baseline_mode}_{args.train_tasks}'
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, group=args.train_tasks[0])
        wandb.config.update(args)

    losses = {}
    timer = {
        "batch_time":AverageMeter('Time', ':6.3f'),
        "data_time": AverageMeter('Data', ':6.3f')
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    best_val_loss = 1000
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch, losses, args, timer, loss_func)
        # print(f"rank {args.rank} has finished the training")
        dist.barrier()
        val_loss = val(val_loader, model, args, epoch)
        if args.wandb_entity is not None and args.rank==0:
            wandb.log({k:v.avg for k,v in losses.items()}, step=epoch)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            train_tasks = "all"
            if args.train_tasks is not None:
                train_tasks = args.train_tasks[0]
            save_name = args.checkpoint_path+'/conv_checkpoint_{}_{}'.format(args.baseline_mode, train_tasks)
            if args.relative:
                save_name += '_relative'
            if args.renew_obs:
                save_name += '_renew'
            if args.add_low_lang:
                save_name += '_low'
            if val_loss<=best_val_loss:
                best_val_loss = val_loss
                save_name_best = save_name + '_best'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.baseline_mode,
                    'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    # 'optimizer' : optimizer.state_dict(),
                    'train_tasks': args.train_tasks
                }, is_best=False, filename=save_name_best)
            if (epoch + 1)%5==0:
                save_name += f"_epoch{epoch}"
                save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.baseline_mode,
                        'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                        # 'optimizer' : optimizer.state_dict(),
                        'train_tasks': args.train_tasks
                    }, is_best=False, filename=save_name)
    if args.wandb_entity is not None and args.rank==0:
        wandb.finish()

def train(data_loader, model, optimizer, scheduler, epoch, losses, args, timer, loss_func):
    if loss_func == None:
        loss_func = torch.nn.BCELoss(reduction='mean')
    batch_time = timer["batch_time"]
    data_time = timer["data_time"]
    model.train()
    bce_loss = torch.nn.BCELoss(reduction='mean')
    end = time.time()
    for batch_step, batch_data in enumerate(data_loader):
        data_time.update(time.time() - end)
        loss_dict = {}
        if len(batch_data)==0:
            continue
        bounds, pixel_size = batch_data[0]['bounds'], batch_data[0]['pixel_size']
        img, language_instructions = [], []
        attention_points, target_points = [], []
        for data in batch_data:
            img.append(data['img'])
            language_instructions += data['language']
            attention_points.append(data['attention_points'])
            target_points.append(data['target_points'])
        img =  np.concatenate(img, axis=0)
        attention_points =  np.concatenate(attention_points, axis=0)
        target_points =  np.concatenate(target_points, axis=0)
        p0 = np.int16((attention_points[:, :2]-bounds[:2,0])/pixel_size)
        p0_z = attention_points[:, 2:3]-bounds[2,0]
        p0_rotation = R.from_quat(attention_points[:, 3:])
        p1 = np.int16((target_points[:, :2]-bounds[:2,0])/pixel_size)
        p1_z = target_points[:, 2:3]-bounds[2,0]
        p1_rotation = R.from_quat(target_points[:, 3:])
        p0 = p0[:,::-1]
        p1 = p1[:,::-1]
        p0_rotation = p0_rotation.as_euler('zyx', degrees=True)
        p1_rotation = p1_rotation.as_euler('zyx', degrees=True)
        inp = {'img':img, 'lang_goal': language_instructions,
            'p0':p0, 'p0_z':p0_z, 'p0_rotation':p0_rotation,
            'p1':p1, 'p1_z':p1_z, 'p1_rotation':p1_rotation}
        loss_dict = model(inp)

        if losses == {}:
            for loss_term in loss_dict:
                losses[loss_term] = AverageMeter(loss_term)
        for loss_term in loss_dict:
            losses[loss_term].update(loss_dict[loss_term].item(), args.batch_size)
        loss = sum(l for l in loss_dict.values())

        optimizer.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        #     if batch_step == 0:
        #         loss.backward(retain_graph=True)
        #     else:
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        batch_time.update(time.time() - end)
        end = time.time()
        # Calculate time remaining.
        time_per_epoch = batch_time.avg * len(data_loader)
        epochs_left = args.epochs - epoch - 1
        batches_left = len(data_loader) - batch_step - 1

        time_left = sec_to_str(batches_left * batch_time.avg + epochs_left * time_per_epoch)
        time_elapsed = sec_to_str(batch_time.sum)
        time_estimate = sec_to_str(args.epochs * time_per_epoch)

        if batch_step % args.log_freq == 0 and args.rank==0:
            tmp_str = 'Epoch: [{}/{}] Batch: [{}/{}]  ' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ' \
                        'Elapsed: {}  ' \
                        'ETA: {} / {}  ' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})  '.format(
                epoch + 1, args.epochs, batch_step, len(data_loader), time_elapsed, time_left, time_estimate,
                batch_time=batch_time, data_time=data_time)
            for loss_term in losses:
                tmp_str += '{}: {loss.val:.4f} ({loss.avg:.4f})  '.format(loss_term, loss=losses[loss_term])
            print(tmp_str)
    
def val(data_loader, model, args, epoch):
    losses= {}
    total_loss = []
    model.eval()
    for batch_step, batch_data in enumerate(data_loader):
        if len(batch_data)==0:
            continue
        bounds, pixel_size = batch_data[0]['bounds'], batch_data[0]['pixel_size']
        img, language_instructions = [], []
        attention_points, target_points = [], []
        for data in batch_data:
            img.append(data['img'])
            language_instructions += data['language']
            attention_points.append(data['attention_points'])
            target_points.append(data['target_points'])
        img =  np.concatenate(img, axis=0)
        attention_points =  np.concatenate(attention_points, axis=0)
        target_points =  np.concatenate(target_points, axis=0)
        if (attention_points[:, :2]>bounds[:2,1]).any() or (attention_points[:, :2]<bounds[:2,0]).any() \
            or (target_points[:, :2]>bounds[:2,1]).any() or (target_points[:, :2]<bounds[:2,0]).any():
            continue
        p0 = np.int16((attention_points[:, :2]-bounds[:2,0])/pixel_size)
        p0_z = attention_points[:, 2:3]-bounds[2,0]
        # p0_theta = R.from_quat(target_points1[:, 3:]).as_euler('zxy', degrees=True)[:, 0]
        # p0_theta = np.int8((p0_theta+180)/10)
        p0_rotation = R.from_quat(attention_points[:, 3:])
        p1 = np.int16((target_points[:, :2]-bounds[:2,0])/pixel_size)
        p1_z = target_points[:, 2:3]-bounds[2,0]
        p1_rotation = R.from_quat(target_points[:, 3:])
        p0 = p0[:,::-1]
        p1 = p1[:,::-1]
        inp = {'img':img, 'lang_goal': language_instructions,
            'p0':p0, 'p0_z':p0_z, 'p0_rotation':p0_rotation.as_euler('zxy', degrees=True),
            'p1':p1, 'p1_z':p1_z, 'p1_rotation':p1_rotation.as_euler('zxy', degrees=True)}
        with torch.no_grad():
            loss_dict = model(inp)
        if losses == {}:
            for loss_term in loss_dict:
                losses[loss_term] = AverageMeter(loss_term)
        for loss_term in loss_dict:
            losses[loss_term].update(loss_dict[loss_term].item(), args.batch_size)
        loss = sum(l.item() for l in loss_dict.values())
        total_loss.append(loss)
    avg_loss = torch.tensor(total_loss).mean(0, keepdim=True).cuda(args.gpu)
    loss_list = [torch.zeros(1).cuda(args.gpu) for _ in range(args.ngpus_per_node)]
    if args.distributed:
        # print(f"rank {args.gpu} has finished the eval!")
        # dist.barrier()
        # dist.all_reduce(avg_loss)
        dist.all_gather(loss_list, avg_loss)
    # avg_loss = total_loss.mean().item()
    if args.rank==0:
        all_avg_loss = torch.tensor(loss_list).mean().item()
        tmp_str = 'Epoch [{}/{}] Val_loss: {:.4f} '.format(epoch + 1, args.epochs, all_avg_loss)
        for loss_term in losses:
            tmp_str += '{}: {loss.val:.4f} ({loss.avg:.4f})  '.format(loss_term, loss=losses[loss_term])
        print(tmp_str)
        return all_avg_loss
    else:
        return None

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    #dataset
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--setd', type=str, default='train')
    parser.add_argument('--img_size',nargs='+', type=int, default=[360, 360])
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--preprocess', action='store_true', 
                help="whether preprocess the data. Next time can directly use. Add if you don't want it.")
    parser.add_argument('--unused_camera_list', nargs='+', default=[None])
    parser.add_argument('--use_fail_cases', action='store_true', help="add if use the fail cases")
    parser.add_argument('--sample_numbers', type=int, default=0, help="downsample from total demonstrations")
    parser.add_argument('--pin_memory', action='store_true', help="do not use if the RAM is small")
    parser.add_argument('--train_tasks', nargs='+', type=str, default = None)
    parser.add_argument('--relative', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--renew_obs', type=lambda x:bool(strtobool(x)), default=True)
    parser.add_argument('--add_low_lang', type=lambda x:bool(strtobool(x)), default=False)
    #traning
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=15, type=int,
                            help='Print log message at this many iterations (default: 10)')
    parser.add_argument('--log-freq', default=1, type=int,
                            help='Print log message at this many iterations (default: 1)')
    parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate for Adam (default: 0.01)')
    parser.add_argument('--checkpoint_path', default='../vlmbench/weights', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume', default= None, type=str,
                        help='resume training from checkpoint-path/model-best.pth')
    parser.add_argument('--baseline_mode', type=str, default='cliport_6dof')
    parser.add_argument('--wandb_entity', type=str, default=None, help="visualize the training. Account Name")
    parser.add_argument('--wandb_project', type=str, default=None,  help="visualize the training. Project Name")

    #distributed training
    parser.add_argument('--world-size', default=1, type=int,
                help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--gpu_number', type=int, default=0)
    parser.add_argument('--gpu_start', type=int, default=0)
    args = parser.parse_args()

    main(args)