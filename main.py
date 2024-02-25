import logging
from pathlib import Path

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
            print("layer name: ",name," mean of the input to the layer: ",m.mean_of_input," a value: ",m.a," osc counter: ",m.osc_counter)

            a_vals=m.a.detach().cpu()
            a_vals=t.reshape(a_vals,(-1,))
            a_vals=a_vals[0:(math.floor(a_vals.numel()**0.5)**2)]
            a_vals=t.reshape(a_vals,(math.floor(a_vals.numel()**0.5),math.floor(a_vals.numel()**0.5)))
            #plt.hist(a_vals)
            #plt.show()
            bar=plt.imshow(a_vals)
            plt.title('a values:')
            plt.colorbar(bar)
            plt.show()
            flip_vals = flip_count[name]
            flip_vals = t.reshape(flip_vals, (-1,))
            flip_vals = flip_vals[0:(math.floor(flip_vals.numel() ** 0.5) ** 2)]
            flip_vals = t.reshape(flip_vals, (math.floor(flip_vals.numel() ** 0.5), math.floor(flip_vals.numel() ** 0.5)))
            # plt.hist(a_vals)
            # plt.show()
            bar = plt.imshow(flip_vals)
            plt.title('flip counter values:')
            plt.colorbar(bar)
            plt.show()
            osc_vals = m.osc_counter.detach().cpu()
            osc_vals = t.reshape(osc_vals, (-1,))
            osc_vals = osc_vals[0:(math.floor(osc_vals.numel() ** 0.5) ** 2)]
            osc_vals = t.reshape(osc_vals, (math.floor(osc_vals.numel() ** 0.5), math.floor(osc_vals.numel() ** 0.5)))
            # plt.hist(a_vals)
            # plt.show()
            bar = plt.imshow(osc_vals)
            plt.title('osc counter values:')
            plt.colorbar(bar)
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
a_opt_lr=1
using_gdtuo=True
def main():
    #print("Regular LSQ, Seed = 0")
    print("getting LSQ SOTA")
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

    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
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

    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)

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
    optim = SGD_for_gSTE(0.01)
    mw = ModuleWrapper(model, optim, modules_to_replace_temp)
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

    # def hook(name, module, grad_input, grad_output):
    #     if name not in cached_grads_alpha:
    #         cached_grads_alpha[name] = []
    #     # Meanwhile store data in the RAM.
    #     # if module.alpha.grad is not None and module.alpha.grad>0:
    #     #    breakpoint()
    #     cached_grads_alpha[name].append(
    #         (1, t.reshape(module.a.data.detach(),(-1,))[0].cpu() ))
    #     # print(name)
    #
    # for name, m in model.named_modules():
    #     if isinstance(m, LsqQuan) and m.is_weight:
    #         print(name)
    #         handlers.append(m.register_full_backward_hook(partial(hook, name)))
    #
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
            t_top1, t_top5, t_loss = process.train(train_loader, mw,using_gdtuo, criterion, optimizer,a_optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, mw, criterion, epoch, monitors, args)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, mw, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
    a_graphs_full(cached_grads_alpha)
    out_graphs(model,flip_count)


if __name__ == "__main__":
    main()
