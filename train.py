# coding=utf-8

# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117


import argparse
import datetime
import os
import traceback

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.rotation_dataset import RotationCocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.Refine_Loss_v2 import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string

# add warmup lr
from torch_warmup_lr import WarmupLR


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('A rotation detector based EfficientDet(Zylo117)')
    parser.add_argument('-p', '--project', type=str, default='rotation_vehicles', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--train_batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('--val_batch_size', type=int, default=8, help='the number of images per batch for val')

    parser.add_argument('--head_only', type=boolean_string, default=True,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--init_head', type=boolean_string, default=True,
                        help='because the rotation detector weights & bias are not the same as original EfficinetDet')

    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=100, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default='weights/efficientdet-d0.pth',
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)  # os.mkdir(path) 只会创建path这个目录，如果上层目录不存在，则报错
    os.makedirs(opt.saved_path, exist_ok=True)  # os.makedirs(path) 会递归创建目录，如果上层目录不存在也会自动创建\
# exist_ok 代表只有目录不存在时才创建，目录已存在不会抛出异常
    training_params = {'batch_size': opt.train_batch_size,
                       'shuffle': True,  # shuffle = True 每个epoch后都会打乱数据集
                       'drop_last': True,  # 处理数据集时，如果数据集长度除于batch_size有余下的数据。True就抛弃，否则保留
                       'collate_fn': collater,  # 如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
                       'num_workers': opt.num_workers}  # 多线程读取数据的方法，设置参数值>1，即可多线程读取数据

    val_params = {'batch_size': opt.val_batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    # training_set = RotationCocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
    #                                    transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
    #                                                                 Augmenter(),
    #                                                                 Resizer(input_sizes[opt.compound_coef])]))

    training_set = RotationCocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                                       transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                                     Resizer(input_sizes[opt.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)
    # DataLoader 定义了Sampler()、BatchSampler()
    # Sampler用于生成每次送入到网络中的(图片)索引indices 按顺序的为SequentialSampler()、乱序的RandomSampler()
    # BatchSampler用于将每次送入到网络中的indices进行打包成一个batch，会根据drop_last参数进行判断
    # 根据BatchSampler中的索引到datasets中寻找图片，送入到模型中

    val_set = RotationCocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                                  transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                                Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        """网络模型保存和加载的两种方法
        1、加载和保存参数
        （1）只保存参数 torch.save(model.state_dict(), path)
        （2）模型加载参数 model.load_state_dict(torch.load(path))
        对于加载参数的这种情况需要注意的是：
        只保存参数的方法在加载的时候要事先定义好跟原模型一致的模型，并在该模型的实例对象(假设名为model)上进行加载；
        即在使用上述加载语句前已经有定义了一个和原模型一样的Net, 并且进行了实例化 model=Net( ) 。
        
        2、保存整个模型（结构+参数）
        （1）保存 torch.save(model, path)
        （2）加载 model = torch.load(path)
        """

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # init head(Regressor & Classifier)
    if opt.init_head:
        def init_head(m):
            classname = m.__class__.__name__
            for ntl in ['regressor', 'classifier']:
                if ntl in classname:
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_normal_(m.weight.data)
                        nn.init.xavier_normal_(m.bias.data)
                        # nn.init.constant_(m.weight.data, 1)
                        # nn.init.constant_(m.bias.data, 0.5)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                        # nn.init.constant_(m.weight, 1)
                        # nn.init.constant_(m.bias, 0)
        model.apply(init_head)
        print('[Info] init head(regressor & classifier)')

    # 检查模型的初始化情况

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batc across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        # 所有层设置相同的学习率
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)

        # 为不同组设置不同的lr
        # classifier_params = list(map(id, model.model.classifier.parameters()))
        # regressor_params = list(map(id, model.model.regressor.parameters()))
        # base_params = filter(lambda p: id(p) not in classifier_params + regressor_params,
        #                      model.parameters())
        # optimizer = torch.optim.AdamW([
        #     {'params': base_params},
        #     {'params': classifier_params, 'lr': opt.lr * 3},
        #     {'params': regressor_params, 'lr': opt.lr * 3}], lr=opt.lr)

    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # modified MultiStepLR
    # 在添加完warmup 后，导致计数有问题，现在的milestone理解为warmup之后，再重新数7个
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.num_epochs * x) for x in [0.7]], gamma=0.1)
    # add warmup lr
    """ torch warmup lr
    (1) init_lr: learning rate will increase from this value to the initialized learning rate 
        in optimizer (in this case 0.01 -> 0.1).
        
    (2) num_warmup: number of steps for warming up learning rate.
    
    (3) warmup_strategy: function that learning rate will gradually increase according to. 
        Currently support cos, linear, constant - learning rate will be fixed 
        and equals to init_lr during warm-up phase).
    """
    # scheduler = WarmupLR(scheduler, init_lr=1e-5, num_warmup=5, warmup_strategy='cos')

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()
    """ model.train()、model.eval()
    model.train() 和 model.eval() 一般在模型训练和评价的时候会加上这两句；
    主要是针对由于model 在训练时和评价时 Batch Normalization 和 Dropout 方法模式不同；
    """

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch  # //整除，向下取整 9 // 2 = 4
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']  # shape(8, 3, 640, 640)
                    annot = data['annot']  # shape(8, 58, 5)

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    # scheduler.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:  # % 取模返回除法的余数
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        # # add begin ------------------------------------------------
                        # torch.save(model, os.path.join(opt.saved_path,
                        #                                f'efficientdet-d{opt.compound_coef}_model_{epoch}_{step}.pth'))
                        # # add end --------------------------------------------
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    # # original
    # if isinstance(model, CustomDataParallel):
    #     torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    # else:
    #     torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))

    # add begin ------------------------------------------
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))
    # add end --------------------------------------------


if __name__ == '__main__':
    opt = get_args()
    train(opt)
