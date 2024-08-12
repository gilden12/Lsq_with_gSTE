import logging
from pathlib import Path
import gc

import torch as t
import yaml

import process
import quan
import util
from model import create_model
from quan.quantizer import lsq
import matplotlib.pyplot as plt
import numpy as np
from quan.quantizer.lsq import *
from functools import partial
import math
import random
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from packaging.version import parse, Version
from util.gdtuo import *

def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.

    Args:
        seed: random seed
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        if parse(tf.__version__) >= Version("2.0.0"):
            tf.random.set_seed(seed)
        elif parse(tf.__version__) <= Version("1.13.2"):
            tf.set_random_seed(seed)
        else:
            tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_random_seed(seed):
    t.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    t.use_deterministic_algorithms(True)
    set_global_seed(seed)
    set_random_seed(seed) 


def out_graphs(model,flip_count):
    y1 = []
    for p in lsq.x:
        y1.append(p)
    x1 = [i for i in range(1, len(y1))]
    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1[1:len(y1)], linestyle='-', label='learned Qp')

    y2 = []
    for p in lsq.thd_list:
        y2.append(p)

    x2 = [i for i in range(1, len(y2) + 1)]
    plt.plot(x2, y2, linestyle='-', label='Normal Qp')
    plt.title('learned Qp vs Normal Qp')  # Set the plot title
    plt.xlabel('Steps')  # Set the X-axis label
    plt.ylabel('Values')  # Set the Y-axis label
    plt.grid(True)  # Add a grid
    plt.legend()
    plt.show()

    for name, m in model.named_modules():
        if isinstance(m, LsqQuan) and m.is_weight:
            #print("layer name: ",name," mean of the input to the layer: ",m.mean_of_input," a value: ",m.a," osc counter: ",m.osc_counter)

            a_vals=m.a.detach().cpu()
            a_vals=t.reshape(a_vals,(-1,))
            a_vals=a_vals[0:(math.floor(a_vals.numel()**0.5)**2)]
            a_vals=t.reshape(a_vals,(math.floor(a_vals.numel()**0.5),math.floor(a_vals.numel()**0.5)))
            #plt.hist(a_vals)
            #plt.show()
            bar=plt.imshow(a_vals)
            plt.title('a v  alues:')
            plt.colorbar(bar)
            plt.show()
            # flip_vals = flip_count[name]
            # flip_vals = t.reshape(flip_vals, (-1,))
            # flip_vals = flip_vals[0:(math.floor(flip_vals.numel() ** 0.5) ** 2)]
            # flip_vals = t.reshape(flip_vals, (math.floor(flip_vals.numel() ** 0.5), math.floor(flip_vals.numel() ** 0.5)))
            # # plt.hist(a_vals)
            # # plt.show()
            # bar = plt.imshow(flip_vals)
            # plt.title('flip counter values:')
            # plt.colorbar(bar)
            # plt.show()
            # osc_vals = m.osc_counter.detach().cpu()
            # osc_vals = t.reshape(osc_vals, (-1,))
            # osc_vals = osc_vals[0:(math.floor(osc_vals.numel() ** 0.5) ** 2)]
            # osc_vals = t.reshape(osc_vals, (math.floor(osc_vals.numel() ** 0.5), math.floor(osc_vals.numel() ** 0.5)))
            # # plt.hist(a_vals)
            # # plt.show()
            # bar = plt.imshow(osc_vals)
            # plt.title('osc counter values:')
            # plt.colorbar(bar)
            # plt.show()

def eta_graphs(eta_dict):
    
    for k,i in eta_dict.items():
        print(torch.flatten(i.detach().cpu()))
        plt.hist(torch.flatten(i.detach().cpu()), bins=60, color='skyblue', edgecolor='black')

        # Adding labels and title
        plt.xlabel('Lr value')
        plt.ylabel('Amount')
        plt.title('Learning rates for a in some layer')
        
        # Display the plot
        plt.show()

def a_graphs_full(alphas_list_full):
    count=0
    lcount=[]
    for x, y in np.ndindex((4, 5)):
        lcount.append([x,y])

    figure, axis = plt.subplots(4, 5)
    for name in alphas_list_full:
        x1 = [i for i in range(1, len(alphas_list_full[name]) + 1)]

        #plt.figure(figsize=(8, 6))
        #plt.plot(x1, alphas_list_full[name], linestyle='-',label= ('Mx Value','Alpha Value'))
        #plt.title('Alpha vs MX '+name)  # Set the plot title
        #plt.xlabel('Steps')  # Set the X-axis label
        #plt.ylabel('Values')  # Set the Y-axis label
        #plt.grid(True)  # Add a grid
        #plt.legend()
        #axis[lcount[count][0], lcount[count][1]].set_title(name)
        axis[lcount[count][0],lcount[count][1]].plot(x1, alphas_list_full[name], linestyle='-',label= ('Normal','learn a'))
        #axis[lcount[count][0], lcount[count][1]].set_title('Alpha vs MX')
        axis[lcount[count][0], lcount[count][1]].legend()
        #axis[lcount[count][0], lcount[count][1]].set_ylim(0.9,1.1)
        count+=1

    plt.show()

    count = 0
    lcount = []
    for x, y in np.ndindex((4, 5)):
        lcount.append([x, y])

    figure, axis = plt.subplots(4, 5)
    for name in alphas_list_full:
        x1 = [i for i in range(1, len(alphas_list_full[name]) + 1)]

        axis[lcount[count][0], lcount[count][1]].set_yscale('log')

        axis[lcount[count][0], lcount[count][1]].plot(x1, alphas_list_full[name], linestyle='-',
                                                      label=('Normal', 'learn a'))
        axis[lcount[count][0], lcount[count][1]].legend()
        #axis[lcount[count][0], lcount[count][1]].set_ylim(0.9,1.1)
        count += 1

    plt.show()
def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)


a_opt_lr=1e4
using_gdtuo=True
#Which solution to use for the problem of depending on the future:
# -1 - regular STE 
# 0 - analytical soltuion for gSTE
# 1 - Delayed Updates
#1.5 Delayed Updates for Meta Network
# 2 - Updating all times together
# 3 - analytical solution for MetaNetwork
# 4 - STE towers
# 5 - analytical soltuion for gSTE in test framework
# 6 - Meta network with delayed updates in test framework
# 7 - Meta network with all times together
# 8 - check baseline which is initial quantization and then full precision training
# 9 - Training all times together with MAD instead of STE
#10 - Training all times together and then taking the last trained a value and keep training with its value without learning a
#11 - Training all times together for x epochs then taking the learned values and training them again all times together to find the following x epochs
num_solution =11
#The learning rate   set used to train the a parameters
a_lr = 0.0
#Decides how many diffrent a parameters for each weight
#0 - a per element, every element in the weight gets a repective a parameter
#1 - a per layer, every weight gets one a parameter
#2 - a per chnnel, every channel of the weight gets a respective a parameter
a_per=0
#if grouping together multiple a values to be a shared parameter to reduce memory consuption this sets the amount of parameter together each time else set 1
num_share_params=1
#In all time todgether training this sets the amount of epochs learned before startin learning from screatch
num_of_epochs_each_time = 1

def main():
    print("Num solution is : ",num_solution)
    #print("Regular LSQ, Seed = 0")
    #Reproducability and comparisons
    seed = 0
    t.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    t.use_deterministic_algorithms(True)
    set_global_seed(seed)
    set_random_seed(seed)
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # Create the model
    model = create_model(args)
    

    T =len(train_loader)*num_of_epochs_each_time# A vector length for All times together solution (# of learning steps before update)
    list_for_lsq=[T, a_per,num_share_params]
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan,num_solution, list_for_lsq)
    modules_to_replace_temp=dict(modules_to_replace)
    model = quan.replace_module_by_names(model, modules_to_replace)
    #tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    logger.info('Inserted quantizers into the original model')

    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)

    model.to(args.device.type)

    start_epoch = 0
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean)

    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device.type)

    for name, param in model.named_parameters():
        print("name parameters : ",name)
    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    
    
    if num_solution == 0:
        main_analyticalgSTE(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif  num_solution == -1:
        main_original(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif num_solution == 1 or num_solution == 1.5:
        main_DelayedUpdates(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif num_solution == 2 or num_solution==7 or num_solution == 8 or num_solution == 9 or num_solution == 10 :
        main_all_times(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time)
    elif num_solution == 5:
        main_analytical_all_time(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T)
    elif num_solution == 6:
        main_DelayedUpdates_meta_network_all_time(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif num_solution == 11:
        main_all_times_repeat(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time)
def main_original(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir):

    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    optimizer = t.optim.SGD(model.parameters(),
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
    t_top1_list=[]
    v_top1_list=[]
    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = process.validate(val_loader, model, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, model, criterion, epoch, monitors, args)
            t_top1_list.append(t_top1)
            v_top1_list.append(v_top1)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
            print("v_top1_list : ",v_top1_list)
            print("t_top1_list : ",t_top1_list)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, model, criterion, -1, monitors, args)

    
    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')

def main_analyticalgSTE(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir):
    a_params_list=[]
    other_params_list=[]
    
    for name, param in model.named_parameters():
        #print("param order check : ",name)
        if name.endswith("a"):
            a_params_list.append(param)
            #print (" params are : ",name, param.data)
        else:
            other_params_list.append(param)
    #to work properly need to remove the s parameter from thd_params
    #thd_parameters = [p for p in model.parameters() if (p.shape[0]==1 and (p.data==3 or p.data==7))]
    #parameters = [p for p in model.parameters() if (p.shape[0]!=1 or (p.shape[0]==1 or p.data==1))]
    #print("t1: ",thd_parameters)
    optimizer = t.optim.SGD(other_params_list,
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    #Here you can change the leraning rate of a
    a_optimizer = t.optim.SGD(a_params_list, lr=a_opt_lr)
    
    
    #gdtuo support
    optim = SGD_for_gSTE(0.01,0.0,1e3)
    mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
    print(" check modules_to_replace_temp : ",modules_to_replace_temp)
    print(" check args.quan.excepts : ",args.quan.excepts)
    mw.initialize()

    thd_optimizer=None

    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)


    cached_grads_alpha = {}
    handlers = []

    #def hook(name, module, grad_input, grad_output):
        #if name not in cached_grads_alpha:
        #    cached_grads_alpha[name] = []
        #Meanwhile store data in the RAM.
        #if module.alpha.grad is not None and module.alpha.grad>0:
        #   breakpoint()
        #cached_grads_alpha[name].append(
        #    (1, 1))
    #    print(name)
    
    #for name, m in model.named_modules():
    #    if isinstance(m, LsqQuan) and m.is_weight:
    #        print(name)
            #handlers.append(m.register_full_backward_hook(partial(hook, name)))
    
    cached_prev_grad = {}
    handlers_grad_flip = []
    flip_count ={}
    # def check_grad_flip_hook(name, module, grad_input, grad_output):
    #     if name not in cached_prev_grad:
    #         cached_prev_grad[name] = t.zeros(1)
    #         flip_count[name] = 0
    #     #print("grad out put : ", grad_output)
    #     if isinstance(grad_output[0],tuple):
    #         #print("in here",grad_output[0])
    #         where_flip=t.where(np.sign(grad_output.cpu()).ne(np.sign(cached_prev_grad[name])),1,0)
    #         flip_count[name]=t.where(where_flip,flip_count[name]+1,0)
    #         cached_prev_grad[name]=grad_output
    #     else:
    #         #print("in there",grad_output[0])
    #
    #         where_flip = t.where(t.sign(grad_output[0].cpu()).ne(t.sign(cached_prev_grad[name].cpu())), 1, 0)
    #         #print(" where met ",where_flip)
    #         flip_count[name] = t.where(where_flip.eq(1), flip_count[name] + 1, flip_count[name])
    #         cached_prev_grad[name] = grad_output[0]
    #
    # for name, m in model.named_modules():
    #     if isinstance(m, LsqQuan) and m.is_weight:
    #         print(name)
    #         handlers_grad_flip.append(m.register_full_backward_hook(partial(check_grad_flip_hook, name)))

    print("len of alphlist = ",len(handlers))

    if not using_gdtuo:
        mw=model
    if args.eval:
        process.validate(test_loader, mw, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = process.validate(val_loader, mw, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process.train_analytical(train_loader, mw,using_gdtuo, criterion, optimizer,a_optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, mw, criterion, epoch, monitors, args)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
    
    print(" check etas : ",mw.optimizer.eta)
    a_graphs_full(cached_grads_alpha)
    eta_graphs(mw.optimizer.eta)
    out_graphs(model,flip_count)


def main_DelayedUpdates(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir):
    a_params_list=[]
    other_params_list=[]
    
    for name, param in model.named_parameters():
        #print("param order check : ",name)
        if name.endswith("a"):
            a_params_list.append(param)
            #print (" params are : ",name, param.data)
        else:
            other_params_list.append(param)
    #to work properly need to remove the s parameter from thd_params
    #thd_parameters = [p for p in model.parameters() if (p.shape[0]==1 and (p.data==3 or p.data==7))]
    #parameters = [p for p in model.parameters() if (p.shape[0]!=1 or (p.shape[0]==1 or p.data==1))]
    #print("t1: ",thd_parameters)
    optimizer = t.optim.SGD(other_params_list,
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    #Here you can change the leraning rate of a
    a_optimizer = t.optim.SGD(a_params_list, lr=a_opt_lr)
    
    
    #gdtuo support
    if num_solution==1:
        optim = SGD_Delayed_Updates(0.01,0.0,1e3)
    else:
        optim = SGD_Delayed_Updates_meta(0.01,0.0,1e3,1.0)

    mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
    print(" check modules_to_replace_temp : ",modules_to_replace_temp)
    print(" check args.quan.excepts : ",args.quan.excepts)
    mw.initialize()

    thd_optimizer=None

    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)


    cached_grads_alpha = {}
    handlers = []

    #def hook(name, module, grad_input, grad_output):
        #if name not in cached_grads_alpha:
        #    cached_grads_alpha[name] = []
        #Meanwhile store data in the RAM.
        #if module.alpha.grad is not None and module.alpha.grad>0:
        #   breakpoint()
        #cached_grads_alpha[name].append(
        #    (1, 1))
    #    print(name)
    
    #for name, m in model.named_modules():
    #    if isinstance(m, LsqQuan) and m.is_weight:
    #        print(name)
            #handlers.append(m.register_full_backward_hook(partial(hook, name)))
    
    cached_prev_grad = {}
    handlers_grad_flip = []
    flip_count ={}
    # def check_grad_flip_hook(name, module, grad_input, grad_output):
    #     if name not in cached_prev_grad:
    #         cached_prev_grad[name] = t.zeros(1)
    #         flip_count[name] = 0
    #     #print("grad out put : ", grad_output)
    #     if isinstance(grad_output[0],tuple):
    #         #print("in here",grad_output[0])
    #         where_flip=t.where(np.sign(grad_output.cpu()).ne(np.sign(cached_prev_grad[name])),1,0)
    #         flip_count[name]=t.where(where_flip,flip_count[name]+1,0)
    #         cached_prev_grad[name]=grad_output
    #     else:
    #         #print("in there",grad_output[0])
    #
    #         where_flip = t.where(t.sign(grad_output[0].cpu()).ne(t.sign(cached_prev_grad[name].cpu())), 1, 0)
    #         #print(" where met ",where_flip)
    #         flip_count[name] = t.where(where_flip.eq(1), flip_count[name] + 1, flip_count[name])
    #         cached_prev_grad[name] = grad_output[0]
    #
    # for name, m in model.named_modules():
    #     if isinstance(m, LsqQuan) and m.is_weight:
    #         print(name)
    #         handlers_grad_flip.append(m.register_full_backward_hook(partial(check_grad_flip_hook, name)))
    
    print("len of alphlist = ",len(handlers))

    if not using_gdtuo:
        mw=model
    if args.eval:
        process.validate(test_loader, mw, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = process.validate(val_loader, mw, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process.train_DelayedUpdates(train_loader, mw,num_solution, criterion, optimizer,a_optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, mw, criterion, epoch, monitors, args)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
    
    print(" check etas : ",mw.optimizer.eta)
    a_graphs_full(cached_grads_alpha)
    eta_graphs(mw.optimizer.eta)
    out_graphs(model,flip_count)

list_train = []
last_train=None
prev_list_train=[]
def main_DelayedUpdates_meta_network_all_time(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir):
    model_copy = copy.deepcopy(model)
    compare_models(model_copy, model)
    train_loader_copy = copy.deepcopy(train_loader)


    a_params_list=[]
    other_params_list=[]
    
    for name, param in model.named_parameters():
        #print("param order check : ",name)
        if name.endswith("a"):
            a_params_list.append(param)
            #print (" params are : ",name, param.data)
        else:
            other_params_list.append(param)
    #to work properly need to remove the s parameter from thd_params
    #thd_parameters = [p for p in model.parameters() if (p.shape[0]==1 and (p.data==3 or p.data==7))]
    #parameters = [p for p in model.parameters() if (p.shape[0]!=1 or (p.shape[0]==1 or p.data==1))]
    #print("t1: ",thd_parameters)
    optimizer = t.optim.SGD(other_params_list,
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    #Here you can change the leraning rate of a
    a_optimizer = t.optim.SGD(a_params_list, lr=a_opt_lr)
    
    optim = SGD_Delayed_Updates_meta_network(0.01,0.0,1e3)

    mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
    print(" check modules_to_replace_temp : ",modules_to_replace_temp)
    print(" check args.quan.excepts : ",args.quan.excepts)
    mw.initialize()

    thd_optimizer=None

    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)


    cached_grads_alpha = {}
    handlers = []
    
    cached_prev_grad = {}
    handlers_grad_flip = []
    flip_count ={}
    
    print("len of alphlist = ",len(handlers))

    if args.eval:
        process.validate(test_loader, mw, criterion, -1, monitors, args)
    else:  # training
        for times in range(start_epoch, args.epochs):
            seed = 0
            set_random_seed(seed)
            #print("cehck 3 :",t.cuda.memory_summary(device=None, abbreviated=False))
            #print("cehck 1 :",t.cuda.memory_summary(device=None, abbreviated=False))
            #train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw)
            train_a_Delayed_updates(times,val_loader,train_loader,start_epoch,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,optimizer,a_optimizer,lr_scheduler)
            #print("cehck 4 :",t.cuda.memory_summary(device=None, abbreviated=False))
            prev_model = model.state_dict()

            #print("cehck 2 :",t.cuda.memory_summary(device=None, abbreviated=False))
            model_new= copy.deepcopy(model_copy)
            with torch.no_grad():#saving trained a values between iterations
                for name, param in model_new.named_parameters():
                    if ("meta_network" in name):
                        
                        param.copy_(prev_model[name].detach())

                        
                        assert torch.equal(model_new.state_dict()[name], prev_model[name])
            #print(t.cuda.memory_summary(device=None, abbreviated=False))
                
            #compare_models(model_copy, model_new)
            model=None
            mw=None
            #torch.cuda.empty_cache()
            model=model_new
            #print("cehck 3 :",t.cuda.memory_summary(device=None, abbreviated=False))

            #print(t.cuda.memory_summary(device=None, abbreviated=False))
            optim = SGD_Delayed_Updates_meta_network(0.01,0.0,1e3)

            mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
            #print("cehck 4 :",t.cuda.memory_summary(device=None, abbreviated=False))

            #print(" check modules_to_replace_temp : ",modules_to_replace_temp)
            #print(" check args.quan.excepts : ",args.quan.excepts)
            mw.initialize()
            
            train_loader=copy.deepcopy(train_loader_copy)
            

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
    
    print_test_graph(list_train)

 

def train_a_Delayed_updates(times,val_loader,train_loader,start_epoch,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,optimizer,a_optimizer,lr_scheduler):
    print(" Starting ",times," time of updating a")
    mw.begin()
    #if args.resume.path or args.pre_trained:
    #        logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
    #        top1, top5, _ = process.validate(val_loader, mw, criterion, start_epoch - 1, monitors, args)
    #        perf_scoreboard.update(top1, top5, start_epoch - 1)
            
    for epoch in range(1):
        
        #print(t.cuda.memory_summary(device=None, abbreviated=False))
        logger.info('>>>>>>>> Epoch %3d' % epoch)
        t_top1, t_top5, t_loss = process.train_DelayedUpdates(train_loader, mw,num_solution, criterion, optimizer,a_optimizer,
                                                   lr_scheduler, epoch, monitors, args)
        v_top1, v_top5, v_loss = process.validate(val_loader, mw, criterion, epoch, monitors, args)
        last_train=[times,t_top1]
        
        tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
        tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
        tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

        perf_scoreboard.update(v_top1, v_top5, epoch)
        is_best = perf_scoreboard.is_best(times)
        #util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)

    mw.zero_grad()

    list_train.append(last_train)
    print("list train is : ",list_train)

def main_analytical_all_time(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T):
    
    model_copy = copy.deepcopy(model)
    compare_models(model_copy, model)
    train_loader_copy = copy.deepcopy(train_loader)
    
    a_params_list=[]
    other_params_list=[]
    
    for name, param in model.named_parameters():
        if name.endswith("a"):
            a_params_list.append(param)
        else:
            other_params_list.append(param)
    optimizer = t.optim.SGD(other_params_list,
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    #Here you can change the leraning rate of a
    a_optimizer = t.optim.SGD(a_params_list, lr=a_opt_lr)
    
    lr_for_a = 1e5
    #gdtuo support
    optim = SGD_for_gSTE(0.01,0.0,lr_for_a)
    mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
    print(" check modules_to_replace_temp : ",modules_to_replace_temp)
    print(" check args.quan.excepts : ",args.quan.excepts)
    mw.initialize()

    thd_optimizer=None

    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)


    cached_grads_alpha = {}
    handlers = []
    cached_prev_grad = {}
    handlers_grad_flip = []
    flip_count ={}
    
    print("len of alphlist = ",len(handlers))




    if not using_gdtuo:
        mw=model
    if args.eval:
        process.validate(test_loader, mw, criterion, -1, monitors, args)
    else:  # training
        for times in range(start_epoch, args.epochs):
            
            seed = 0
            t.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False

            t.use_deterministic_algorithms(True)
            set_global_seed(seed)
            set_random_seed(seed)  
            #print("cehck 3 :",t.cuda.memory_summary(device=None, abbreviated=False))
            #print("cehck 1 :",t.cuda.memory_summary(device=None, abbreviated=False))
            #train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw)
            train_a_analytical(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,optimizer,a_optimizer,lr_scheduler)
            #print("cehck 4 :",t.cuda.memory_summary(device=None, abbreviated=False))
            prev_model = model.state_dict()

            #print("cehck 2 :",t.cuda.memory_summary(device=None, abbreviated=False))
            model_new= copy.deepcopy(model_copy)
            with torch.no_grad():#saving trained a values between iterations
                for name, param in model_new.named_parameters():
                    if name.endswith('.a'):
                        
                        param.copy_(prev_model[name].detach())

                        
                        assert torch.equal(model_new.state_dict()[name], prev_model[name])
            #print(t.cuda.memory_summary(device=None, abbreviated=False))
                
            #compare_models(model_copy, model_new)
            model=None
            mw=None
            #torch.cuda.empty_cache()
            model=model_new
            #print("cehck 3 :",t.cuda.memory_summary(device=None, abbreviated=False))

            #print(t.cuda.memory_summary(device=None, abbreviated=False))
            optim = SGD_for_gSTE(0.01,0.0,1e3)
            mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)
            #print("cehck 4 :",t.cuda.memory_summary(device=None, abbreviated=False))

            #print(" check modules_to_replace_temp : ",modules_to_replace_temp)
            #print(" check args.quan.excepts : ",args.quan.excepts)
            mw.initialize()
            
            train_loader=copy.deepcopy(train_loader_copy)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
    
    print_test_graph(list_train)

def train_a_analytical(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,optimizer,a_optimizer,lr_scheduler):
    print(" Starting ",times," time of updating a")
    mw.begin()
    #if args.resume.path or args.pre_trained:
    #        logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
    #        top1, top5, _ = process.validate(val_loader, mw, criterion, start_epoch - 1, monitors, args)
    #        perf_scoreboard.update(top1, top5, start_epoch - 1)
            
    for epoch in range(1):
        
        #print(t.cuda.memory_summary(device=None, abbreviated=False))
        logger.info('>>>>>>>> Epoch %3d' % epoch)
        t_top1, t_top5, t_loss = process.train_analytical(train_loader, mw,using_gdtuo, criterion, optimizer,a_optimizer,
                                                   lr_scheduler, epoch, monitors, args)
        v_top1, v_top5, v_loss = process.validate(val_loader, mw, criterion, epoch, monitors, args)
        last_train=[times,t_top1]
        
        tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
        tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
        tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

        perf_scoreboard.update(v_top1, v_top5, epoch)
        is_best = perf_scoreboard.is_best(times)
        #util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)

    mw.zero_grad()

    list_train.append(last_train)
    print("list train is : ",list_train)

def main_all_times_repeat(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time):
    torch.backends.cudnn.deterministic = True
    
    v_top1_list=[]
    if num_solution == 11:
        for seg in range(0,500):
            print(seg," Segment of training, using the optimal weights we found for previous segment")
            #if seg != 0:
            #    model.load_state_dict(torch.load('/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth"))       
            model_copy = copy.deepcopy(model)

            
            optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
            mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
            #print(" check modules_to_replace_temp : ",modules_to_replace_temp)
            #print(" check args.quan.excepts : ",args.quan.excepts)
            mw.initialize()

            perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
            counter=0
            
            if args.eval:
                process.validate(test_loader, mw, criterion, -1, monitors, args)
            else:  # training
                for times in range(start_epoch, args.epochs):
                    print(times ," time of finding the optimal weights for this segment")
                    seed = 0
                    t.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed)
                    t.cuda.manual_seed_all(seed)
                    t.backends.cudnn.deterministic = True
                    t.backends.cudnn.benchmark = False

                    t.use_deterministic_algorithms(True)
                    set_global_seed(seed)
                    set_random_seed(seed)  
                    #temp_copy= copy.deepcopy(model)
                    #print("beg train_a in main :",t.cuda.memory_summary(device=None, abbreviated=False))

                    train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,seg)
                    v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
                    v_top1_list.append(v_top1)
                    torch.save(model.state_dict(), '/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth")
                    
                    prev_model = model.state_dict()
                    
                    #print(t.cuda.memory_summary(device=None, abbreviated=False))
                    #print("bef model_new :",t.cuda.memory_summary(device=None, abbreviated=False))
                    model_new= copy.deepcopy(model_copy)
                    #print("aft model_new :",t.cuda.memory_summary(device=None, abbreviated=False))
                    
                    
                    with torch.no_grad():#saving trained a values between iterations
                        flag=0
                        
                        for name, param in model_new.named_parameters():
                            if counter ==args.epochs-1:
                                #print("Im here dslhnsdklncvskdjvnc")
                                param.copy_(prev_model[name])
                            else:
                                if name.endswith('.a'):
                                    #print("check if grad is not zero : ",param.grad)
                                    param.copy_(prev_model[name])
                                    assert torch.equal(model_new.state_dict()[name], prev_model[name])
                                    #print("print histogram value : ",prev_model[name])
                                    #print("print name : ",name)
                                    if flag==1:
                                        
                                        tensor_histogram(param.cpu())
                                    flag+=1
                    counter+=1
                    print("aft val assignment :",t.cuda.memory_summary(device=None, abbreviated=False))

                    model=None
                    model=model_new
                    
                    
                    if num_solution == 7:
                        optim = SGD_Delayed_Updates_meta_network(0.01,0.0,a_lr)
                    else:
                        optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
                    
                    mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)

                    mw.initialize()
                    #print("aft end of loop :",t.cuda.memory_summary(device=None, abbreviated=False))
                    gc.collect()
                    torch.cuda.empty_cache()
                    #train_loader=copy.deepcopy(train_loader_copy)
                    print("v_top1_list : ",v_top1_list)

                    #print("aft end of loop :",t.cuda.memory_summary(device=None, abbreviated=False))

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')

def main_all_times_keep_last_trained_a(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time):
    torch.backends.cudnn.deterministic = True
    
            
    if num_solution == 10:
        print(seg," Segment of training, using the optimal weights we found for previous segment")
        #if seg != 0:
        #    model.load_state_dict(torch.load('/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth"))       
        model_copy = copy.deepcopy(model)

        
        optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
        mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
        #print(" check modules_to_replace_temp : ",modules_to_replace_temp)
        #print(" check args.quan.excepts : ",args.quan.excepts)
        mw.initialize()

        perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
        counter=0
        if args.eval:
            process.validate(test_loader, mw, criterion, -1, monitors, args)
        else:  # training
            
            for times in range(start_epoch, args.epochs):
                print(times ," time of finding the optimal weights for this segment")
                seed = 0
                t.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                t.cuda.manual_seed_all(seed)
                t.backends.cudnn.deterministic = True
                t.backends.cudnn.benchmark = False

                t.use_deterministic_algorithms(True)
                set_global_seed(seed)
                set_random_seed(seed)  
                #temp_copy= copy.deepcopy(model)
                #print("beg train_a in main :",t.cuda.memory_summary(device=None, abbreviated=False))

                train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,0)
                torch.save(model.state_dict(), '/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth")
                
                prev_model = model.state_dict()
                
                #print(t.cuda.memory_summary(device=None, abbreviated=False))
                #print("bef model_new :",t.cuda.memory_summary(device=None, abbreviated=False))
                model_new= copy.deepcopy(model_copy)
                #print("aft model_new :",t.cuda.memory_summary(device=None, abbreviated=False))
                
                
                with torch.no_grad():#saving trained a values between iterations
                    flag=0
                    for name, param in model_new.named_parameters():
                        if counter ==args.epochs-1:
                            #print("Im here dslhnsdklncvskdjvnc")
                            param.copy_(prev_model[name])
                        if name.endswith('.a'):
                            #print("check if grad is not zero : ",param.grad)
                            param.copy_(prev_model[name])
                            assert torch.equal(model_new.state_dict()[name], prev_model[name])
                            #print("print histogram value : ",prev_model[name])
                            #print("print name : ",name)
                            if flag==1:
                                
                                tensor_histogram(param.cpu())
                            flag+=1
                counter+=1
                print("aft val assignment :",t.cuda.memory_summary(device=None, abbreviated=False))


                if times==args.epochs-1:
                    model.load_state_dict(torch.load('/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth"))       
                    for name, mod in model.named_modules():
                        if name.endswith('quan_w_fn'):
                            #print("check if grad is not zero : ",param.grad)
                            mod.use_last_a_trained()
                            #assert torch.equal(model_new.state_dict()[name], prev_model[name])
                            #print("print histogram value : ",prev_model[name])
                            #print("print name : ",name)
    
                    a_lr=0
                    train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,100,0)

                model=None
                model=model_new
                
                
                if num_solution == 7:
                    optim = SGD_Delayed_Updates_meta_network(0.01,0.0,a_lr)
                else:
                    optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
                
                mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)

                mw.initialize()
                #print("aft end of loop :",t.cuda.memory_summary(device=None, abbreviated=False))
                gc.collect()
                torch.cuda.empty_cache()
                #train_loader=copy.deepcopy(train_loader_copy)
                
                #print("aft end of loop :",t.cuda.memory_summary(device=None, abbreviated=False))

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')
            


    print_test_graph(list_train)

    print(" check etas : ",mw.optimizer.eta)
    eta_graphs(mw.optimizer.eta)
def main_all_times(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time):
    torch.backends.cudnn.deterministic = True
    
    model_copy = copy.deepcopy(model)
    compare_models(model_copy, model)
    org_mod=model
    model = model_copy
    model_copy = org_mod
    #train_loader_copy = copy.deepcopy(train_loader)


    #gdtuo support
    #a_lr = 1e3
    if num_solution == 7:
        optim = SGD_Delayed_Updates_meta_network(0.01,0.0,a_lr)
    else:
        optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
    mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
    #print(" check modules_to_replace_temp : ",modules_to_replace_temp)
    #print(" check args.quan.excepts : ",args.quan.excepts)
    mw.initialize()

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
    counter=0
    v_top1_list=[]
    if args.eval:
        process.validate(test_loader, mw, criterion, -1, monitors, args)
    else:  # training
        for times in range(start_epoch, args.epochs):
            
            seed = 0
            t.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False

            t.use_deterministic_algorithms(True)
            set_global_seed(seed)
            set_random_seed(seed)  
            #temp_copy= copy.deepcopy(model)
            #print("beg train_a in main :",t.cuda.memory_summary(device=None, abbreviated=False))

            train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,0)
            v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
            v_top1_list.append(v_top1)
            torch.save(model.state_dict(), '/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth")
            
            prev_model = model.state_dict()
            
            #print(t.cuda.memory_summary(device=None, abbreviated=False))
            #print("bef model_new :",t.cuda.memory_summary(device=None, abbreviated=False))
            model_new= copy.deepcopy(model_copy)
            #print("aft model_new :",t.cuda.memory_summary(device=None, abbreviated=False))
            
            if num_solution == 7:
                with torch.no_grad():#saving trained a values between iterations
                    for name, param in model_new.named_parameters():
                        if ("meta_modules" in name):
                            
                            param.copy_(prev_model[name].detach())

                            
                            assert torch.equal(model_new.state_dict()[name], prev_model[name])
            else:
                if num_solution == 8:
                    with torch.no_grad():#saving trained a values between iterations
                        for name, param in model_new.named_parameters():
                            #if name.endswith('.x_hat'):
                            param.copy_(prev_model[name])
                            assert torch.equal(model_new.state_dict()[name], prev_model[name])
                else:
                    with torch.no_grad():#saving trained a values between iterations
                        flag=0
                        for name, param in model_new.named_parameters():
                            if name.endswith('.a'):
                                #print("check if grad is not zero : ",param.grad)
                                param.copy_(prev_model[name])
                                assert torch.equal(model_new.state_dict()[name], prev_model[name])
                                #print("print histogram value : ",prev_model[name])
                                #print("print name : ",name)
                                if flag==1:
                                    
                                    tensor_histogram(param.cpu())
                                flag+=1
            
            #print("aft val assignment :",t.cuda.memory_summary(device=None, abbreviated=False))

            model=None
            model=model_new
            
            
            #if num_solution == 7:
            #    optim = SGD_Delayed_Updates_meta_network(0.01,0.0,a_lr)
            #else:
            #    optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
            
            #mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)

            #mw.initialize()
            #print("aft end of loop :",t.cuda.memory_summary(device=None, abbreviated=False))
            gc.collect()
            torch.cuda.empty_cache()
            #train_loader=copy.deepcopy(train_loader_copy)
            
            #print("aft end of loop :",t.cuda.memory_summary(device=None, abbreviated=False))
            print("v_top1_list : ",v_top1_list)
            # counter+=1
            # if counter == 15:
            #     a_lr=1e1

        #finished training
          
            
        

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')
        logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
            


    print_test_graph(list_train)

    print(" check etas : ",mw.optimizer.eta)
    eta_graphs(mw.optimizer.eta)

def print_test_graph(data):
    # Separate the data into x and y values
    x_values = [point[0] for point in data]
    y_values = [point[1] for point in data]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Updates for a')
    plt.ylabel('Training accuracy')


    # Show the plot
    plt.grid(True)
    plt.show()


def train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,base):
    print(" Starting ",times," time of updating a")
    #print("bef begin :",t.cuda.memory_summary(device=None, abbreviated=False))
    mw.begin()
    #print("aft begin :",t.cuda.memory_summary(device=None, abbreviated=False))

    if num_solution==7:
        mw.zero_grad_meta()
    else:
        mw.zero_grad()
    
    this_trining_list=[]
    count=0
    #print("bef for loop train :",t.cuda.memory_summary(device=None, abbreviated=False))
    prev_last=[]
    last_train=[]
    for epoch in range(num_of_epochs_each_time):
        
        logger.info('>>>>>>>> Epoch %3d' % (base*num_of_epochs_each_time+epoch))
        #print("bef train ",t.cuda.memory_summary(device=None, abbreviated=False))
        t_top1, t_top5, t_loss = process.train_all_times(train_loader, mw,num_solution,T, criterion, epoch, monitors, args,base*num_of_epochs_each_time)
        #print("aft train",t.cuda.memory_summary(device=None, abbreviated=False))
        #v_top1, v_top5, v_loss = process.validate(val_loader, mw, criterion, epoch, monitors, args)
        prev_last=last_train
        last_train=[times,t_top1]

        this_trining_list.append(t_top1)
        count+=1
        #tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
        #tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
        #tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

        #perf_scoreboard.update(v_top1, v_top5, epoch)
        #is_best = perf_scoreboard.is_best(times)
        #util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
        if num_solution==7:
            mw.step_meta()
        else:
            if T/len(train_loader) == count:
                print("check")
                mw.step_a()
                
        
        if num_solution==7:
            mw.zero_grad_meta()
        else:
            if T/len(train_loader) == count:
                mw.zero_grad()
                count=0
        
        #print(t.cuda.memory_summary(device=None, abbreviated=False))
        #mw.zero_grad()
        #gc.collect()
        #torch.cuda.empty_cache()
        #print(t.cuda.memory_summary(device=None, abbreviated=False))
    print("Current : ",this_trining_list)
    mw =None
    prev_list_train.append(prev_last)
    print("prev list train is : ",prev_list_train)
    list_train.append(last_train)
    print("list train is : ",list_train)
        

def tensor_histogram(tensor):
    # Convert the tensor to a numpy array
    plt.close('all')

    tensor_np = tensor.numpy()
    
    # Flatten the tensor to 1D for the histogram
    tensor_np_flat = tensor_np.flatten()
    
    # Plot the histogram
    plt.ion()  # Enable interactive mode
    plt.hist(tensor_np_flat, bins=100,range=(0,5), edgecolor='black')
    plt.title('Histogram of Tensor Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.draw()
    plt.pause(0.001)  # Allow the plot to update
    
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

if __name__ == "__main__":
    main()
