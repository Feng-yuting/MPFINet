import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
import os
from torch.autograd import Variable
from PIL import Image
import cv2
from model import *
from config import *
from dataset import FlameSet,DataPrefetcher
def main():
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print('======>Load datasets')
    train_set = FlameSet(opt.dataset_train)
    val_set = FlameSet(opt.dataset_val)
    train_data_loader = DataLoader(dataset=train_set,batch_size=opt.batchSize,shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = MPFINet()

    criterion = nn.L1Loss()
    min_loss = 100.

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        print("===> Setting GPU  Cuda")

    print("===> Setting Optimizer and lr_scheduler")
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr,betas=[0.9,0.99])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[301,601],gamma=0.1)
    print("===> Parameter numbers : %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("===> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict((checkpoint["optimizer_state_dict"]))
        else:
            print("===> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("===> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("===> no model found at '{}'".format(opt.pretrained))


    print("===> Training")
    t = time.strftime("%Y%m%d%H%M")
    if not os.path.exists('log'):
        os.mkdir('log')
    global log
    log = 'log/'+t+ '_batchSize%d' % (opt.batchSize)+'_lr%f'% (opt.lr)+'_QB'
    if not os.path.exists(log):
        os.mkdir(log)
    backup_model = os.path.join(log,opt.backup)
    if not os.path.exists(backup_model):
        os.mkdir(backup_model)#备份pkl文件
    best_model = os.path.join(log,opt.best_model)
    if not os.path.exists(best_model):
        os.mkdir(best_model)#存储最好结果pkl


    train(train_data_loader,val_data_loader, optimizer, model, scheduler,criterion,t,backup_model,best_model,min_loss)

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10"""
#     lr = opt.lr * (0.1 ** (epoch // 300))
#     return lr

def train(train_data_loader,val_data_loader, optimizer, model, scheduler,criterion,t,backup_model,best_model,min_loss):

    print('===> Begin Training!')
    model.train()
    steps_per_epoch = len(train_data_loader)
    if opt.train_log:
        if os.path.isfile(opt.train_log):
            train_log = open(opt.train_log,"a")  # log/Networkname_202111022144_train.log
        else:
            print("=> no train_log found at '{}'".format(opt.train_log))
    else:
            train_log = open(os.path.join(log, "%s_%s_train.log") % (opt.net, t),"w")  # log/Networkname_202111022144_train.log

    if opt.epoch_time_log:
        if os.path.isfile(opt.epoch_time_log):
            epoch_time_log = open(opt.epoch_time_log,"a")
        else:
            print("==> no epoch_time_log found at '{}'".format(opt.epoch_time_log))
    else:
        epoch_time_log = open(os.path.join(log, "%s_%s_epoch_time.log") % (opt.net, t), "w")
    time_sum = 0

    global val_log
    if opt.val_log:
        if os.path.isfile(opt.val_log):
            val_log = open(opt.val_log, "a")
        else:
            print("===> no val_log found at '{}'".format(opt.val_log))
    else:
        val_log = open(os.path.join(log, "%s_%s_val.log") % (opt.net, t), "w")

    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        # 计时,每个epoch的时间
        start = time.time()

        #加快读取数据集
        prefetcher = DataPrefetcher(train_data_loader)
        data = prefetcher.next()

        i = 0
        while data is not None:
            # run_step()
            i += 1
            if i >= steps_per_epoch:
                break  # 循环次数从0开始计数还是从1开始计数的问题
            input_pan, input_ms, target = data[0], data[1],data[2]
            if opt.cuda:
                input_pan = input_pan.cuda()
                input_ms = input_ms.cuda()
                target = target.cuda()
            output = model(input_pan, input_ms)
            train_loss = criterion(output, target)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        scheduler.step()#调整lr
        print("===> Epoch[{}/{}]: Lr:{} Train_Loss: {:.10f} ".format(epoch,opt.nEpochs,optimizer.param_groups[0]["lr"],train_loss.item()))
        train_log.write("Epoch[{}/{}]: Train_Loss: {:.15f}\n".format(epoch, opt.nEpochs, train_loss.item()))

        if epoch % 50 ==0:
            save_checkpoint(model, epoch, optimizer, backup_model)
        # backup a model every ** epochs and validate
        if epoch % 2 == 0:
            val_retloss = val(val_data_loader, model, criterion, epoch)
            val_log.write("{} {:.10f}\n".format((epoch), val_retloss))

            # Save the best weight
            if min_loss > val_retloss:
                save_bestmodel(model, epoch, best_model)
                min_loss = val_retloss

        # 输出每轮训练花费时间
        time_epoch = (time.time() - start)
        time_sum += time_epoch

        epoch_time_log.write("No:{} epoch training costs {:.4f}min\n".format(epoch, time_epoch / 60))

    train_log.close()
    val_log.close()
    epoch_time_log.close()

def val(val_data_loader, model, criterion, epoch):
    model.eval()
    avg_l1 = 0
    val_loss=[]
    with torch.no_grad():
        for k, data in enumerate(val_data_loader):
            input_pan, input_ms, target = data[0], data[1],data[2]
            if opt.cuda:
                input_pan = input_pan.cuda()
                input_ms = input_ms.cuda()
                target = target.cuda()

            output = model(input_pan, input_ms)
            loss = criterion(output, target)
            avg_l1 += loss.item()

    del (input_pan, input_ms,target, output)
    print("===> Epoch{} Avg. Val Loss: {:.10f}".format(epoch, avg_l1 / len(val_data_loader)))#ssim_value
    val_loss.append(avg_l1)
    return avg_l1 / len(val_data_loader)


def save_checkpoint(model, epoch,optimizer,backup_model):
    model_out_path = os.path.join(backup_model,"model_epoch_last.pth".format(epoch))
    state = {"epoch": epoch, "model_state_dict": model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}
    torch.save(state, model_out_path)
    print("===> Checkpoint saved to {}".format(epoch))


def save_bestmodel(model, epoch,backupbest_model):
    model_out_path = os.path.join(backupbest_model,"model_best_epoch.pth")
    state = {"epoch": epoch, "model": model}
    torch.save(state, model_out_path)
    print("Save best model at {} epoch!!!".format(epoch))



if __name__ == "__main__":
    main()






