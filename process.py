import logging
import math
import operator
import time

import torch as t

from util import AverageMeter
import main
__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

a_optimizer_lr=main.a_opt_lr

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device.type)
        targets = targets.to(args.device.type)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr']
                })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


def train_analytical(train_loader, model,using_gdtuo, criterion, optimizer,a_optimizer, lr_scheduler, epoch, monitors, args):
    if using_gdtuo:
        mw=model
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #print("cehck 4 :",t.cuda.memory_summary(device=None, abbreviated=False))
        #print(t.cuda.memory_summary(device=None, abbreviated=False))
        for name, module in model.named_modules():
            #if "conv" in name:
            #    module.get_v_hat_grads=True
            module.get_v_hat_grads=True
        if using_gdtuo:
            mw.begin()
        inputs = inputs.to(args.device.type)
        targets = targets.to(args.device.type)


        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)

        # is_a_warmup=1
        #
        # if epoch==0 and is_a_warmup==1:
        #     n=600
        #     if batch_idx <=n:
        #         for g in a_optimizer.param_groups:
        #             g['lr'] = (batch_idx) *(1/(n))*a_optimizer_lr
        #
        # if epoch==0 and is_a_warmup==2:
        #     n=200
        #     if batch_idx <=n:
        #         for g in a_optimizer.param_groups:
        #             g['lr'] = (batch_idx** 2) *(1/(n ** 2))*a_optimizer_lr
        #
        
        if using_gdtuo:
            if not( (epoch==0) and (batch_idx==0)):
                mw.zero_grad()
                loss.backward(create_graph=True)
                mw.get_dl_dv_hat()
                mw.step_a()
                
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #No need for step with gdtuo yet just got gradients for v_hat


        #a_optimizer.step()
        
        for name, module in model.named_modules():
            #if "conv" in name:
            #    module.get_v_hat_grads=False
            module.get_v_hat_grads=False
        if using_gdtuo:
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            mw.zero_grad()
            #mw.check_grad_vals()

            loss.backward(create_graph=True) # important! use create_graph=True
            #print("calling step")
            mw.step_w()
            
        
        if False:
            mw.zero_grad()
            loss.backward(create_graph=True) # important! use create_graph=True
            mw.step()
            print("finished time")

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr']
                })
    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg

def train_DelayedUpdates(train_loader, model,num_solution, criterion, optimizer,a_optimizer, lr_scheduler, epoch, monitors, args):
    
    mw=model

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    step_counter = 0
    prev_grads = {}
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #print("beggining of batch loop : ",t.cuda.memory_summary(device=None, abbreviated=False))
        mw.begin()
        inputs = inputs.to(args.device.type)
        targets = targets.to(args.device.type)


        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)

        
        if num_solution ==6:

            mw.zero_grad()
            loss.backward(create_graph=True)
            mw.step_w()

            if train_DelayedUpdates.counter % 2 ==1:
                mw.step_meta()
            
            #print("cehck after step_a in proccess : ",t.cuda.memory_summary(device=None, abbreviated=False))
    

        else:
            if train_DelayedUpdates.counter % 2 ==1:
                mw.zero_grad()
            
                loss.backward(create_graph=True)
                mw.step_w()

            if train_DelayedUpdates.counter % 2 ==0:
                mw.zero_grad()
                loss.backward(create_graph=True)
                mw.step_w()
                if not( (epoch==0) and (batch_idx==0)):
                    if num_solution==1.5:
                        mw.step_a_and_b()
                    else:
                        print("trying to sstep a")
                        mw.step_a()

        if not( (epoch==0) and (batch_idx==0)):
            train_DelayedUpdates.counter+=1
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr']
                })
        #print("end of batch loop : ",t.cuda.memory_summary(device=None, abbreviated=False))


    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg
train_DelayedUpdates.counter=0



def train_all_times(train_loader, model,num_solution,T, criterion, epoch, monitors, args):
    mw=model

    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    #model.eval()
    end_time = time.time()
    step_counter = 0
    prev_grads = {}
    counter = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #print(t.cuda.memory_summary(device=None, abbreviated=False))
        if num_solution == 7:
            mw.begin_w_meta()
        else:
            mw.begin_w()
        
        inputs = inputs.to(args.device.type)
        targets = targets.to(args.device.type)


        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if counter != 0 and counter%T ==0:
            
            if num_solution == 7:
                mw.step_meta()
                mw.zero_grad_meta()
                mw.begin()# Is this cheating? Isnt it just hiding some problem? we should not need to do that
                mw.zero_grad_meta()
            else:
                mw.step_a()
                mw.zero_grad()
        #if batch_idx !=0:
        #    #print("check what zeros : ")
        #    mw.step_w()
        if num_solution == 7:
            mw.zero_grad_not_meta()
        else:
            mw.zero_grad_not_a()
        #if batch_idx !=0:
        #    print("check afterrrrrr what zeros : ")
        #    mw.step_w()
        loss.backward(create_graph=True)
            

        mw.step_w()
            
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': 0,
                })
        counter+=1

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, criterion, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with t.no_grad():
            inputs = inputs.to(args.device.type)
            targets = targets.to(args.device.type)

            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.log.print_freq == 0:
                for m in monitors:
                    m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
