import cv2
import os
import sys
import time
import numpy as np
import datetime
import logging
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import nyudv2
from models import Model

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)

def main():
    model_name = '3dgnn_enet'
    current_path = os.getcwd()
    logger = logging.getLogger(model_name)
    log_path = current_path + '/artifacts/'+ str(datetime.datetime.now().strftime('%Y-%m-%d-%H')).replace(' ', '/') + '/'
    print('log path is:',log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(log_path + 'save/')
    hdlr = logging.FileHandler(log_path + model_name + '.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("Loading data...")
    print("Loading data...")

    label_to_idx = {'<UNK>': 0, 'beam': 1, 'board': 2, 'bookcase': 3, 'ceiling': 4, 'chair': 5, 'clutter': 6,
                    'column': 7,
                    'door': 8, 'floor': 9, 'sofa': 10, 'table': 11, 'wall': 12, 'window': 13}
    idx_to_label = {0: '<UNK>', 1: 'beam', 2: 'board', 3: 'bookcase', 4: 'ceiling', 5: 'chair', 6: 'clutter',
                    7: 'column',
                    8: 'door', 9: 'floor', 10: 'sofa', 11: 'table', 12: 'wall', 13: 'window'}

    '''Data Loader parameter'''
    # Batch size
    batch_size_tr = 4
    batch_size_va = 4
    # Multiple threads loading data
    workers_tr = 4
    workers_va = 4
    # Data augmentation
    flip_prob = 0.5
    crop_size = 0

    dataset_tr = nyudv2.Dataset(flip_prob=flip_prob,crop_type='Random',crop_size=crop_size)
    dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size_tr, shuffle=True,
                               num_workers=workers_tr, drop_last=False, pin_memory=True)

    dataset_va = nyudv2.Dataset(flip_prob=0.0,crop_type='Center',crop_size=crop_size)
    dataloader_va = DataLoader(dataset_va, batch_size=batch_size_va, shuffle=False,
                               num_workers=workers_va, drop_last=False, pin_memory=True)
    cv2.setNumThreads(workers_tr)

    class_weights = [0.0]+[1.0 for i in range(13)]
    nclasses = len(class_weights)
    num_epochs = 50

    '''GNN parameter'''
    use_gnn = True
    gnn_iterations = 3
    gnn_k = 64
    mlp_num_layers = 1

    '''Model parameter'''
    use_bootstrap_loss = False
    bootstrap_rate = 0.25
    use_gpu = True

    logger.info("Preparing model...")
    print("Preparing model...")
    model = Model(nclasses, mlp_num_layers,use_gpu)
    loss = nn.NLLLoss(reduce=not use_bootstrap_loss, weight=torch.FloatTensor(class_weights))
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)

    if use_gpu:
        model = model.cuda()
        loss = loss.cuda()
        softmax = softmax.cuda()
        log_softmax = log_softmax.cuda()

    '''Optimizer parameter'''
    base_initial_lr = 5e-4
    gnn_initial_lr = 1e-3
    betas = [0.9, 0.999]
    eps = 1e-08
    weight_decay = 1e-4
    lr_schedule_type = 'exp'
    lr_decay = 0.9
    lr_patience = 10

    optimizer = torch.optim.Adam([{'params': model.decoder.parameters()},
                                  {'params': model.gnn.parameters(), 'lr': gnn_initial_lr}],
                                 lr=base_initial_lr, betas=betas, eps=eps, weight_decay=weight_decay)

    if lr_schedule_type == 'exp':
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / num_epochs)), lr_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif lr_schedule_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay, patience=lr_patience)
    else:
        print('bad scheduler')
        exit(1)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Number of trainable parameters: %d", params)

    def get_current_learning_rates():
        learning_rates = []
        for param_group in optimizer.param_groups:
            learning_rates.append(param_group['lr'])
        return learning_rates

    def eval_set(dataloader):
        model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            confusion_matrix = torch.cuda.FloatTensor(np.zeros(14 ** 2))

            start_time = time.time()

            for batch_idx, rgbd_label_xy in enumerate(dataloader):

                sys.stdout.write('\rEvaluating test set... {}/{}'.format(batch_idx + 1, len(dataloader)))
                x = rgbd_label_xy[0]
                xy = rgbd_label_xy[2]
                target = rgbd_label_xy[1].long()
                x = x.float()
                xy = xy.float()

                input = x.permute(0, 3, 1, 2).contiguous()
                xy = xy.permute(0, 3, 1, 2).contiguous()
                if use_gpu:
                    input = input.cuda()
                    xy = xy.cuda()
                    target = target.cuda()

                output = model(input, gnn_iterations=gnn_iterations, k=gnn_k, xy=xy, use_gnn=use_gnn)

                if use_bootstrap_loss:
                    loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                    topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                               int((crop_size ** 2) * bootstrap_rate))
                    loss_ = torch.mean(topk)
                else:
                    loss_ = loss.forward(log_softmax(output.float()), target)
                loss_sum += loss_

                pred = output.permute(0, 2, 3, 1).contiguous()
                pred = pred.view(-1, nclasses)
                pred = softmax(pred)
                pred_max_val, pred_arg_max = pred.max(1)

                pairs = target.view(-1) * 14 + pred_arg_max.view(-1)
                for i in range(14 ** 2):
                    cumu = pairs.eq(i).float().sum()
                    confusion_matrix[i] += cumu.item()

            sys.stdout.write(" - Eval time: {:.2f}s \n".format(time.time() - start_time))
            loss_sum /= len(dataloader)

            confusion_matrix = confusion_matrix.cpu().numpy().reshape((14, 14))
            class_iou = np.zeros(14)
            # we ignore void values
            confusion_matrix[0, :] = np.zeros(14)
            confusion_matrix[:, 0] = np.zeros(14)
            for i in range(1, 14):
                class_iou[i] = confusion_matrix[i, i] / (
                        np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i])

        return loss_sum.item(), class_iou, confusion_matrix

    '''Training parameter'''
    model_to_load = None
    logger.info("num_epochs: %d", num_epochs)
    print("Number of epochs: %d"%num_epochs)
    interval_to_show = 100

    train_losses = []
    eval_losses = []

    if model_to_load:
        logger.info("Loading old model...")
        print("Loading old model...")
        model.load_state_dict(torch.load(model_to_load))
    else:
        logger.info("Starting training from scratch...")
        print("Starting training from scratch...")

    '''Training'''
    for epoch in range(1, num_epochs + 1):
        batch_loss_avg = 0
        if lr_schedule_type == 'exp':
            scheduler.step(epoch)
        for batch_idx, rgbd_label_xy in enumerate(dataloader_tr):

            sys.stdout.write('\rTraining data set... {}/{}'.format(batch_idx + 1, len(dataloader_tr)))

            x = rgbd_label_xy[0]
            target = rgbd_label_xy[1].long()
            xy = rgbd_label_xy[2]
            x = x.float()
            xy = xy.float()

            input = x.permute(0, 3, 1, 2).contiguous()
            input = input.type(torch.FloatTensor)

            if use_gpu:
                input = input.cuda()
                xy = xy.cuda()
                target = target.cuda()

            xy = xy.permute(0, 3, 1, 2).contiguous()

            optimizer.zero_grad()
            model.train()

            output = model(input, gnn_iterations=gnn_iterations, k=gnn_k, xy=xy, use_gnn=use_gnn)

            if use_bootstrap_loss:
                loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                           int((crop_size ** 2) * bootstrap_rate))
                loss_ = torch.mean(topk)
            else:
                loss_ = loss.forward(log_softmax(output.float()), target)

            loss_.backward()
            optimizer.step()

            batch_loss_avg += loss_.item()

            if batch_idx % interval_to_show == 0 and batch_idx > 0:
                batch_loss_avg /= interval_to_show
                train_losses.append(batch_loss_avg)
                logger.info("E%dB%d Batch loss average: %s", epoch, batch_idx, batch_loss_avg)
                print('\rEpoch:{}, Batch:{}, loss average:{}'.format(epoch, batch_idx, batch_loss_avg))
                batch_loss_avg = 0

        batch_idx = len(dataloader_tr)
        logger.info("E%dB%d Saving model...", epoch, batch_idx)

        torch.save(model.state_dict(),log_path +'/save/'+'checkpoint_'+str(epoch)+'.pth')

        '''Evaluation'''
        eval_loss, class_iou, confusion_matrix = eval_set(dataloader_va)
        eval_losses.append(eval_loss)

        if lr_schedule_type == 'plateau':
            scheduler.step(eval_loss)
        print('Learning ...')
        logger.info("E%dB%d Def learning rate: %s", epoch, batch_idx, get_current_learning_rates()[0])
        print('Epoch{} Def learning rate: {}'.format(epoch, get_current_learning_rates()[0]))
        logger.info("E%dB%d GNN learning rate: %s", epoch, batch_idx, get_current_learning_rates()[1])
        print('Epoch{} GNN learning rate: {}'.format(epoch, get_current_learning_rates()[1]))
        logger.info("E%dB%d Eval loss: %s", epoch, batch_idx, eval_loss)
        print('Epoch{} Eval loss: {}'.format(epoch, eval_loss))
        logger.info("E%dB%d Class IoU:", epoch, batch_idx)
        print('Epoch{} Class IoU:'.format(epoch))
        for cl in range(14):
            logger.info("%+10s: %-10s" % (idx_to_label[cl], class_iou[cl]))
            print('{}:{}'.format(idx_to_label[cl], class_iou[cl]))
        logger.info("Mean IoU: %s", np.mean(class_iou[1:]))
        print("Mean IoU: %.2f"%np.mean(class_iou[1:]))
        logger.info("E%dB%d Confusion matrix:", epoch, batch_idx)
        logger.info(confusion_matrix)


    logger.info("Finished training!")
    logger.info("Saving model...")
    print('Saving final model...')
    torch.save(model.state_dict(), log_path + '/save/3dgnn_enet_finish.pth')
    eval_loss, class_iou, confusion_matrix = eval_set(dataloader_va)
    logger.info("Eval loss: %s", eval_loss)
    logger.info("Class IoU:")
    for cl in range(14):
        logger.info("%+10s: %-10s" % (idx_to_label[cl], class_iou[cl]))
    logger.info("Mean IoU: %s", np.mean(class_iou[1:]))


if __name__ == '__main__':
    main()
