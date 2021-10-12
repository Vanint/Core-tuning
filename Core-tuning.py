'''
The source codes of Contrast-regularized Tuning on CIFAR-10
Copyright (c) 2021: Unleashing the Power of Contrastive Self-Supervised Visual Models via Contrast-Regularized Fine-Tuning (NeurIPS21) 
'''
from __future__ import print_function

import argparse
import os
import PIL
import shutil
import time
import random
import numpy as np
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import models.imagenet as customized_models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
 
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names
model_names.append('resnet50-sl')
model_names.append('resnet50-ssl')

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Cifar-10 Training')
parser.add_argument('-d', '--dataset', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)') 
parser.add_argument('--eta_weight', default=10.0, type=float, metavar='N',
                    help='trade-off parameter')
parser.add_argument('--contrast_dim', type=int, default=256, help='Contrastive feature dimensions.')  
parser.add_argument('--mixup_alpha', type=float, default=1, help='alpha for mixup')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[41, 81],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50-ssl',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)') 
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES') 
parser.add_argument('--accumulate_step',  type=int, default= 1)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


class ResnetSSLModel(nn.Module):
    def __init__(self, num_class):
        super(ResnetSSLModel, self).__init__()
        dim_feature = 2048   
        num_classes =  num_class
        self.extractor = models.__dict__['resnet50'](num_classes=num_classes)   
        self.extractor.load_state_dict(torch.load("./checkpoint/pretrain_moco_v2.pkl"),strict=False)     
        del self.extractor.fc
        self.extractor.fc=lambda x:x
        self.classifier = nn.Linear(dim_feature, num_classes, bias=False)
        self.contrastor = Contrastor(dim_feature, args.contrast_dim) 
  
    def forward(self, x, target_1hot= None, mixup = False):
        batch_size = x.shape[0] 
        features = self.extractor(x)
        if mixup:
            augment_features, augment_target_1hot, contrastive_features, contrastive_target_1hot=  feature_mixup(features, target_1hot, batch_size)
            augment_contrastor_features = self.contrastor(contrastive_features)
            augment_prediciton = self.classifier(augment_features) 
            prediction  = self.classifier(features) 
            return prediction.cuda(), augment_features.cuda(), augment_contrastor_features.cuda(), augment_prediciton.cuda(), augment_target_1hot.cuda(), contrastive_target_1hot.cuda()
        else:
            contrastor_features = self.contrastor(features)
            prediction  = self.classifier(features) 
            return features.cuda(), contrastor_features.cuda(), prediction.cuda()
 
class Contrastor(nn.Module): 
    def __init__(self, n_inputs, n_outputs):
        super(Contrastor, self).__init__()
        self.input = nn.Linear(n_inputs, n_inputs) 
        self.output = nn.Linear(n_inputs, n_outputs) 
    def forward(self, x):
        x = self.input(x) 
        x = F.relu(x) 
        x = self.output(x)
        return F.normalize(x)


def to_one_hot(label, num_classes):
    y_onehot = torch.FloatTensor(label.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, label.unsqueeze(1).cpu(), 1)
    return y_onehot.to("cuda")

def feature_mixup(features, targets, batch_size):
    one_hot_labels =  to_one_hot(targets,args.num_classes)
    list_f = features.clone().detach().requires_grad_(True)
    list_y = one_hot_labels.clone().detach().requires_grad_(True) 
    batch_size = features.shape[0]
    feature_norm = F.normalize(features)
    targets =targets.contiguous().view(-1, 1)
    equal_mask = torch.eq(targets, targets.T).float().cuda()
    logits_mask = torch.scatter(torch.ones_like(equal_mask), 1, torch.arange(batch_size).view(-1, 1).cuda(), 0)    
    positive_mask = equal_mask * logits_mask
    negative_mask = 1-equal_mask
    similarity_matrix = torch.matmul(feature_norm, feature_norm.T)
    positive_matrix = similarity_matrix * positive_mask
    negative_matrix = similarity_matrix * negative_mask 
     
    # hard negative pair generation. 
    order = np.arange(batch_size) 
    random.shuffle(order)   
    # We randomly shuffle the sample order to find negative pairs. Even if a few pairs are not negative here, it won't affect the overall functionality. 
    features2 = features[order]    
    y_2 = one_hot_labels[order]
    lam = np.random.beta(args.mixup_alpha, args.mixup_alpha,batch_size)
    for i in range(batch_size): 
        if lam[i] <= 0.8:
            lam[i] = 0.8 
    lam = torch.from_numpy(lam).view([batch_size,-1]).float().cuda()   # BS,1  
    f_ = (1 - lam) * features +  lam * features2
    y_ = (1 - lam) * one_hot_labels + lam * y_2
    list_f = torch.cat((list_f,f_), dim=0)
    list_y = torch.cat((list_y,y_), dim=0) 

    # hard positive pair generation 
    hard_negative_index = negative_matrix.argmax(dim=1)
    hard_negative_features = features[hard_negative_index] 
    hard_negative_y  = one_hot_labels[hard_negative_index]
    hard_positive_index = positive_matrix.argmin(dim=1)
    for i in range(batch_size):  
        if hard_positive_index[i] == 0:
            hard_positive_index[i] = i 
    hard_positive_features = features[hard_positive_index]   
    hard_positive_y = one_hot_labels[hard_positive_index]   
    lam = np.random.beta(args.mixup_alpha, args.mixup_alpha,batch_size) 
    lam = torch.from_numpy(lam).view([batch_size,-1]).float().cuda()          
    f_ = lam * hard_positive_features + (1 - lam) * hard_negative_features   
    y_ = lam * hard_positive_y + (1 - lam) * hard_negative_y   
    list_f = torch.cat((list_f,f_), dim=0)
    list_y = torch.cat((list_y,y_), dim=0)    
    contrastive_f = list_f    
    contrastive_y = list_y
       
    return list_f.to("cuda"), list_y.to("cuda"), contrastive_f.to("cuda"), contrastive_y.to("cuda")
 
    
def focal_SupConLoss(features, labels=None, mask=None, mixup=False):
    """
    Partial codes are based on the implementation of supervised contrastive loss. 
    import from https https://github.com/HobbitLong/SupContrast.
    """
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))
    temperature=0.07
    base_temperature=0.07
    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        if mixup:
            if labels.size(1)>1:
                weight_index = 10**np.arange(args.num_classes)  
                weight_index = torch.tensor(weight_index).unsqueeze(1).to("cuda")
                labels_ = labels.mm(weight_index.float()).squeeze(1)
                labels_ = labels_.detach().cpu().numpy()
                le = preprocessing.LabelEncoder()
                le.fit(labels_)
                labels = le.transform(labels_)
                labels=torch.unsqueeze(torch.tensor(labels),1)
        labels = labels.contiguous().view(-1, 1) 
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)
   
    anchor_feature = features.float()  
    contrast_feature = features.float()
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),temperature)  
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()  
    logits_mask = torch.scatter(
        torch.ones_like(mask),  
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask   

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
    
    # compute weight
    weight = 1-torch.exp(log_prob)
    
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (weight * mask * log_prob).mean(1)

    # loss
    mean_log_prob_pos = - (temperature / base_temperature) * mean_log_prob_pos
    mean_log_prob_pos = mean_log_prob_pos.view(batch_size)
    
    N_nonzeor = torch.nonzero(mask.sum(1)).shape[0]
    loss = mean_log_prob_pos.sum()/N_nonzeor
    if torch.isnan(loss):
         print("nan contrastive loss")
         loss=torch.zeros(1).to(device)          
    return loss




def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset) 
    transform_train = transforms.Compose([
        transforms.Resize((272,272), interpolation=PIL.Image.BICUBIC),
        transforms.RandomRotation(15,),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
     
    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=PIL.Image.BICUBIC),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
    args.num_classes= num_classes

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers) 
    
    # Model
    print("==> creating model '{}'".format(args.arch))
    # create model 
    dim_feature = 2048    
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    elif args.arch == 'resnet50-sl':
        model = models.resnet50(pretrained=True) 
        model.fc = nn.Linear(dim_feature, num_classes)
    elif args.arch == 'resnet50-ssl':
        model = ResnetSSLModel(num_classes) 
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
 
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) 

    # resume
    title = 'Cifar10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and validation
    for epoch in range(start_epoch, args.epochs): 
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
             logger.write("The best performance:" + str(best_acc))         
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, best_acc, checkpoint=args.checkpoint)
        # adjust learning rate
        scheduler.step()        
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train() 
    accumulate_step = args.accumulate_step
    batch_time = AverageMeter()
    data_time = AverageMeter() 
    classification_losses = AverageMeter()
    contrastive_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets) 
        
        # compute output
        outputs, augment_features, contrast_features,  augment_outputs, augment_target_1hot, contrastive_target_1hot  = model(inputs, targets, mixup=True)
        
        new_size = augment_features.shape[0]  
        all_logits = F.log_softmax(augment_outputs, dim=1)
        classification_loss = -torch.sum(all_logits * augment_target_1hot, dim=1)
        classification_loss = classification_loss.view(new_size).mean()
        
        contrastive_loss = focal_SupConLoss(contrast_features, contrastive_target_1hot, mixup=True)
        total_loss = classification_loss + args.eta_weight * contrastive_loss

        # measure accuracy and record loss
        features, contrast_features,  outputs  = model(inputs, mixup=False)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        classification_losses.update(classification_loss.item(), inputs.size(0))
        contrastive_losses.update(contrastive_loss.item(), inputs.size(0))
        losses.update(total_loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step. Accumulate losses for larger batch size. When accumulate=1, the standard SGD
        total_loss = total_loss/accumulate_step
        total_loss.backward()
        if ((batch_idx+1)%accumulate_step)==0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Classification_Loss: {classification_loss:.4f} |  Contrast_Loss: {contrastive_loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    classification_loss=classification_losses.avg,
                    contrastive_loss = contrastive_losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    torch.cuda.empty_cache()    
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    classification_losses = AverageMeter()
    contrastive_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        features, contrast_features,  outputs  = model(inputs, mixup=False)
        classification_loss = criterion(outputs, targets)
        contrastive_loss = focal_SupConLoss(contrast_features, targets)
        total_loss = classification_loss + args.eta_weight * contrastive_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        classification_losses.update(classification_loss.item(), inputs.size(0))
        contrastive_losses.update(contrastive_loss.item(), inputs.size(0))
        losses.update(total_loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Classification_Loss: {classification_loss:.4f} |  Contrast_Loss: {contrastive_loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    classification_loss=classification_losses.avg,
                    contrastive_loss = contrastive_losses.avg,
                    top1=top1.avg,
                    )
       
        bar.next()
    bar.finish()
    torch.cuda.empty_cache()    
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, best_acc, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        print("The best performance:", best_acc)
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

