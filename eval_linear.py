# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from sklearn.metrics import confusion_matrix, f1_score

import utils
import vision_transformer as vits
import dataproc_double as dp


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # print(embed_dim)
    linear_classifier = LinearClassifier(1, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============
    # same mean and std for pre and post images
    mean_pre = (0.39327543, 0.40631564, 0.32678495)
    std_pre = (0.16512179, 0.14379614, 0.15171282)
    mean_post = (0.39327543, 0.40631564, 0.32678495)
    std_post = (0.16512179, 0.14379614, 0.15171282)

    val_transform = pth_transforms.Compose([
        dp.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        dp.ToTensor(),
        dp.Normalize_Std(mean_pre, std_pre, mean_post, std_post)
    ])

    dataset_val = dp.xBD_Building_Polygon_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_valid,
                                    transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = pth_transforms.Compose([
        dp.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        dp.RandomFlip(p=0.5),
        dp.ToTensor(),
        dp.Normalize_Std(mean_pre, std_pre, mean_post, std_post)
    ])

    dataset_train = dp.xBD_Building_Polygon_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_train,
                                    transform=train_transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train)

    # data balance
    # sampler = torch.utils.data.WeightedRandomSampler(dataset_train.class_weights, num_samples=len(4))

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, torch.from_numpy(dataset_train.class_weights))
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}")
            print(f"Confusion Matrix at epoch {epoch}: \n {test_stats['cm']}")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            print(f"F1 score at epoch {epoch}: {test_stats['f1']:.3f}")
        if utils.is_main_process():
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool, class_weights=None):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for it, batch in enumerate(metric_logger.log_every(loader, 20, header)):
        bldg_pre = batch['bldg_pre']
        bldg_post = batch['bldg_post']
        bldg_label = batch['label']
        batch_size = bldg_post.shape[0]
        # print("Batch size: ", batch_size) # 128

        # move to gpu
        bldg_pre = bldg_pre.float().cuda(non_blocking=True) 
        bldg_post = bldg_post.float().cuda(non_blocking=True) 
        bldg_label = bldg_label.cuda(non_blocking=True) 
        class_weights = class_weights.float().cuda(non_blocking=True)

        # inp = inp.cuda(non_blocking=True)       # input data
        # target = target.cuda(non_blocking=True) # target label

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output_pre = model.get_intermediate_layers(bldg_pre, n)
                output_pre = torch.cat([x[:, 0] for x in intermediate_output_pre], dim=-1)
                intermediate_output_post = model.get_intermediate_layers(bldg_post, n)
                output_post = torch.cat([x[:, 0] for x in intermediate_output_post], dim=-1)
                if avgpool:
                    output_pre = torch.cat((output_pre.unsqueeze(-1), torch.mean(intermediate_output_pre[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output_pre = output_pre.reshape(output_pre.shape[0], -1)
                    output_post = torch.cat((output_post.unsqueeze(-1), torch.mean(intermediate_output_post[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output_post = output_post.reshape(output_post.shape[0], -1)
            else:
                output_pre = model(bldg_pre)
                output_post = model(bldg_post)
        # Calculate cosine similarity
        output_pre = F.normalize(output_pre, p=2, dim=1)
        output_post = F.normalize(output_post, p=2, dim=1)  # [128, 1536]
        # output = torch.linalg.norm(output_pre-output_post, dim=1) # [128, 1]

        # Calculate cosine similarity
        output = F.cosine_similarity(output_pre, output_post, dim=1)

        # train the linear classifier
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss(weight=class_weights)(output, bldg_label)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    cm = np.zeros((4, 4))
    for it, batch in enumerate(metric_logger.log_every(val_loader, 20, header)):
        bldg_pre = batch['bldg_pre']
        bldg_post = batch['bldg_post']
        bldg_label = batch['label']

        # move to gpu
        bldg_pre = bldg_pre.float().cuda(non_blocking=True) 
        bldg_post = bldg_post.float().cuda(non_blocking=True) 
        bldg_label = bldg_label.cuda(non_blocking=True) 

        # inp = inp.cuda(non_blocking=True)       # input data
        # target = target.cuda(non_blocking=True) # target label

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output_pre = model.get_intermediate_layers(bldg_pre, n)
                output_pre = torch.cat([x[:, 0] for x in intermediate_output_pre], dim=-1)
                intermediate_output_post = model.get_intermediate_layers(bldg_post, n)
                output_post = torch.cat([x[:, 0] for x in intermediate_output_post], dim=-1)
                if avgpool:
                    output_pre = torch.cat((output_pre.unsqueeze(-1), torch.mean(intermediate_output_pre[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output_pre = output_pre.reshape(output_pre.shape[0], -1)
                    output_post = torch.cat((output_post.unsqueeze(-1), torch.mean(intermediate_output_post[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output_post = output_post.reshape(output_post.shape[0], -1)
            else:
                output_pre = model(bldg_pre)
                output_post = model(bldg_post)
                
        output_pre = F.normalize(output_pre, p=2, dim=1)
        output_post = F.normalize(output_post, p=2, dim=1)
        # Calculate l2 similarity
        # output = torch.linalg.norm(output_pre-output_post, dim=1) # [128, 1]

        # Calculate cosine similarity
        output = F.cosine_similarity(output_pre, output_post, dim=1) # [128, 1]

        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, bldg_label)

        _, predicted = torch.max(output, 1)

        # Convert tensors to CPU and to NumPy arrays
        predicted_np = predicted.cpu().numpy()
        labels_np = bldg_label.cpu().numpy()

        # Calculate confusion matrix
        current_cm = confusion_matrix(labels_np, predicted_np)
        if cm is None:
            cm = current_cm
        else:
            cm += current_cm
        # Calculate F1 score
        f1 = f1_score(labels_np, predicted_np, average='weighted')

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, bldg_label, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, bldg_label, topk=(1,))

        batch_size = bldg_pre.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()}, 'cm': cm, 'f1': f1}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', type=str, default= "/home/bpeng/mnt/mnt242/scdm_data/xBD/xbd_disasters_building_polygons_neighbors")
    parser.add_argument('--csv_train', type=str, default='csvs_buffer/sub_valid_wo_unclassified.csv', help='train csv sub-path within data path')
    parser.add_argument('--csv_valid', type=str, default='csvs_buffer/sub_valid_wo_unclassified.csv', help='valid csv sub-path within data path')
    # all data: train_tier3_test_hold_wo_unclassified
    # train data: sub_train_wo_unclassified
    # valid data: sub_valid_wo_unclassified
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./linear", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=4, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    eval_linear(args)
