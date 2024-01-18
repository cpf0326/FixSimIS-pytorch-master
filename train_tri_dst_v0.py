import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tllib.self_training.dst import ImageClassifier, WorstCaseEstimationLoss
from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
import utils.get_model
from utils.pseudo_label import ConfidenceBasedSelfTrainingLoss

logger = logging.getLogger(__name__)
best_acc = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=50, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--tri_mergin', default=1.0, type=float,
                        help='tri mergin')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lambda_tri', default=0.1, type=float,
                        help='coefficient of tri loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.7, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    # dst
    # model parameters
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=utils.get_model.get_model_names(),
    #                     help='backbone architecture: ' + ' | '.join(utils.get_model.get_model_names()) + ' (default: resnet50)')
    parser.add_argument('--width', default=2048, type=int,
                        help='width of the pseudo head and the worst-case estimation head')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true', default=False,
                        help='no pool layer after the feature extractor')
    parser.add_argument('--pretrained-backbone', default=None, type=str,
                        help="pretrained checkpoint of the backbone "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='whether to use 10x smaller lr for backbone')


    # training parameters
    parser.add_argument('--eta', default=1, type=float,
                        help='the trade-off hyper-parameter of adversarial loss')
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    parser.add_argument('--eta-prime', default=2, type=float,
                        help="the trade-off hyper-parameter between adversarial loss on labeled data "
                             "and that on unlabeled data")

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        import models.wideresnet_dst as models
        backbone = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes)
        # backbone = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone)
        num_classes = args.num_classes
        pool_layer = nn.Identity() if args.no_pool else None
        classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, width=args.width,
                                         pool_layer=pool_layer, finetune=args.finetune).to(device)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in classifier.parameters())/1e6))
        return classifier

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)



def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        # 平均处理器
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        wce_losses = AverageMeter()
        losses_tri = AverageMeter()
        mask_probs = AverageMeter()
        self_training_criterion = ConfidenceBasedSelfTrainingLoss(args.threshold).to(device)
        worst_case_estimation_criterion = WorstCaseEstimationLoss(args.eta_prime).to(device)
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            # 算法step2
            try:
                inputs_x_w, inputs_x_s, targets_x = labeled_iter.next()
                x_l = inputs_x_w.to(args.device)
                x_l_strong = inputs_x_s.to(args.device)
                labels_l =targets_x.to(args.device)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x_w, inputs_x_s, targets_x = labeled_iter.next()
                x_l = inputs_x_w.to(args.device)
                x_l_strong = inputs_x_s.to(args.device)
                labels_l = targets_x.to(args.device)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            # 算法3-4
            try:
                # 添加 inputs_or
                (inputs_u_w, inputs_u_s, inputs_or), _ = unlabeled_iter.next()
                x_u = inputs_u_w.to(args.device)
                x_u_strong = inputs_u_s.to(args.device)
                inputs_or = inputs_or.to(args.device)
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                # 添加inputs_or
                (inputs_u_w, inputs_u_s, inputs_or), _ = unlabeled_iter.next()
                x_u = inputs_u_w.to(args.device)
                x_u_strong = inputs_u_s.to(args.device)
                inputs_or = inputs_or.to(args.device)
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_u_w.shape[0]
            # 带标签的数据一共有B笔，无标签数据一共有μB笔(每笔3张图像)，加起来就是(3μ+1)B笔
            y_l_strong, _, _, _ = model(x_l_strong)
            cls_loss_strong = args.trade_off_cls_strong * F.cross_entropy(y_l_strong, labels_l)
            cls_loss_strong.backward()
            x = torch.cat((x_l, x_u), dim=0)
            outputs, outputs_adv, _, _ = model(x)
            # inputs = interleave(
            #     torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_or)), 3*args.mu+1).to(args.device)
            y_l, y_u = outputs.chunk(2, dim=0)
            y_l_adv, y_u_adv = outputs_adv.chunk(2, dim=0)
            # print(inputs)
            # tri = de_interleave(tri, 3 * args.mu + 1)
            # _ = tri[:batch_size]  # 前B笔
            # tri_u_w, tri_u_s, tri_or = tri[batch_size:].chunk(3)  # 拆分

            # del logits, outputs_adv, outputs_adv_u_strong  # 省出内存
            # 带标签数据的损失函数
            # Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            # ==============================================================================================================
            # cross entropy loss (weak augment)
            # ==============================================================================================================
            cls_loss_weak = F.cross_entropy(y_l, labels_l)
            # ==============================================================================================================
            # worst case estimation loss
            # ==============================================================================================================
            wce_loss = args.eta * worst_case_estimation_criterion(y_l, y_l_adv, y_u, y_u_adv)
            (cls_loss_weak + wce_loss).backward()

            # ==============================================================================================================
            # self training loss
            # ==============================================================================================================
            # # 比0.95大才说明这个标签置信度高，如果低于这个阈值，即使计算了交叉熵，也会被mask为0
            # pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            # mask = max_probs.ge(args.threshold).float()
            #
            # # Lu ,mask, pseudo_labels = self_training_criterion(logits_u_s, logits_u_w)
            # Lu = (F.cross_entropy(y_u_strong, targets_u,
            #                       reduction='none') * mask).mean()
            # ==============================================================================================================
            # self training loss
            # ==============================================================================================================
            _, _, y_u_strong = model(x_u_strong)
            self_training_loss, mask, pseudo_labels = self_training_criterion(y_u_strong, y_u)
            self_training_loss = args.trade_off_self_training * self_training_loss
            self_training_loss.backward()
            # # 添加三元组损失
            # # triplet_loss = nn.TripletMarginLoss(reduction='none')
            # # margin 暂时设置成1，越大logits_or，logits_u_w越近
            # triplet_loss=torch.nn.TripletMarginLoss(margin=args.tri_mergin, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None,
            #                            reduction='mean')
            # trip_loss = triplet_loss(tri_or, tri_u_w, tri_u_s)
            # pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            # mask = max_probs.ge(args.threshold).float()

            # measure accuracy and record loss
            cls_loss = cls_loss_strong + cls_loss_weak
            cls_losses.update(cls_loss.item(), batch_size)
            loss = cls_loss + self_training_loss + wce_loss
            losses.update(loss.item(), batch_size)
            wce_losses.update(wce_loss.item(), batch_size)
            self_training_losses.update(self_training_loss.item(), batch_size)

            cls_acc = accuracy(y_l, labels_l)[0]
            cls_accs.update(cls_acc.item(), batch_size)

            # ratio of pseudo labels
            n_pseudo_labels = mask.sum()
            ratio = n_pseudo_labels / batch_size
            pseudo_label_ratios.update(ratio.item() * 100, batch_size)

            # accuracy of pseudo labels
            if n_pseudo_labels > 0:
                pseudo_labels = pseudo_labels * mask - (1 - mask)
                n_correct = (pseudo_labels == labels_u).float().sum()
                pseudo_label_acc = n_correct / n_pseudo_labels * 100
                pseudo_label_accs.update(pseudo_label_acc.item(), n_pseudo_labels)

            # 添加三元组损失后的总损失，超参数暂时设置0.1，先写死
            # loss = Lx + args.lambda_u * Lu+args.lambda_tri*trip_loss+args.eta *wce_loss
            loss = Lx + args.lambda_u * Lu + args.lambda_tri * trip_loss
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_tri.update(trip_loss)
            # wce_losses.update(wce_loss.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_tri: {losses_tri:.4f}.  Wce_losses: {wce_losses:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    losses_tri=losses_tri.avg,
                    wce_losses=wce_losses.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.triple_loss', wce_losses.avg, epoch)
            args.writer.add_scalar('train/5.w', test_loss, epoch)
            args.writer.add_scalar('train/6.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)


            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
