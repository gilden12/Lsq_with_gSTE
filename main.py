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
num_solution =8
#The learning rate   set used to train the a parameters
a_lr = 0.0
#Decides how many diffrent a parameters for each weight
#0 - a per element, every element in the weight gets a repective a parameter
#1 - a per layer, every weight gets one a parameter
#2 - a per chnnel, every channel of the weight gets a respective a parameter
a_per=1
#if grouping together multiple a values to be a shared parameter to reduce memory consuption this sets the amount of parameter together each time else set 1
num_share_params=1
#In all time todgether training this sets the amount of epochs learned before startin learning from screatch
num_of_epochs_each_time = 200

def main():
    print("Num solution is : ",num_solution)
    seed = 0
    t.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    t.use_deterministic_algorithms(True)
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

    if num_solution == 0:
        pass
    #    main_analyticalgSTE(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif  num_solution == -1:
        main_original(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    #elif num_solution == 1 or num_solution == 1.5:
    #    main_DelayedUpdates(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif num_solution == 2 or num_solution==7 or num_solution == 8 or num_solution == 9 or num_solution == 10 :
        main_all_times(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time)
    #elif num_solution == 5:
    #    main_analytical_all_time(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T)
    #elif num_solution == 6:
    #    main_DelayedUpdates_meta_network_all_time(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
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


list_train = []
last_train=None
prev_list_train=[]

def main_all_times_repeat(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time):
    torch.backends.cudnn.deterministic = True
    
    v_top1_list=[]
    if num_solution == 11:
        for seg in range(0,500):
            print(seg," Segment of training, using the optimal weights we found for previous segment")
            model_copy = copy.deepcopy(model)

            
            optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
            mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
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
                    set_random_seed(seed)  

                    train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,seg)
                    v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
                    v_top1_list.append(v_top1)
                    torch.save(model.state_dict(), '/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth")
                    
                    prev_model = model.state_dict()
                    
                    model_new= copy.deepcopy(model_copy)
                    
                    
                    with torch.no_grad():#saving trained a values between iterations
                        flag=0
                        
                        for name, param in model_new.named_parameters():
                            if counter ==args.epochs-1:
                                param.copy_(prev_model[name])
                            else:
                                if name.endswith('.a'):
                                    param.copy_(prev_model[name])
                                    assert torch.equal(model_new.state_dict()[name], prev_model[name])
                                    if flag==1:
                                        
                                        tensor_histogram(param.cpu())
                                    flag+=1
                    counter+=1
                    print("aft val assignment :",t.cuda.memory_summary(device=None, abbreviated=False))

                    model=None
                    model=model_new
                    
                    
                    optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
                    
                    mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)

                    mw.initialize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("v_top1_list : ",v_top1_list)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')

def main_all_times(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time):
    torch.backends.cudnn.deterministic = True
    
    model_copy = copy.deepcopy(model)
    compare_models(model_copy, model)

    optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
    mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
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
            set_random_seed(seed)  
            v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)

            train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,0)
            v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
            v_top1_list.append(v_top1)
            torch.save(model.state_dict(), '/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth")
            
            prev_model = model.state_dict()
            
            model_new= copy.deepcopy(model_copy)
            
            
            
            if num_solution == 8:
                mw.detach_params()
                model_new.load_state_dict(prev_model)
                #with torch.no_grad():#saving trained a values between iterations
                #    for name, param in model_new.named_parameters():
                #        #if name.endswith('.x_hat'):
                #        param.copy_(prev_model[name])
                #        assert torch.equal(model_new.state_dict()[name], prev_model[name])
            else:
                with torch.no_grad():#saving trained a values between iterations
                    flag=0
                    for name, param in model_new.named_parameters():
                        if name.endswith('.a'):
                            param.copy_(prev_model[name])
                            assert torch.equal(model_new.state_dict()[name], prev_model[name])
                            if flag==1:
                                tensor_histogram(param.cpu())
                            flag+=1
            
            model=None
            model=model_new
            
            
            
            optim = SGD_Delayed_Updates(0.01,0.0,a_lr)
            
            mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)

            mw.initialize()

            v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
            gc.collect()
            torch.cuda.empty_cache()
            
            print("v_top1_list : ",v_top1_list)          
            
        

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')
        logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
            




def train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,base):
    print(" Starting ",times," time of updating a")
    mw.begin()

    mw.zero_grad()
    
    this_training_list=[]
    count=0
    prev_last=[]
    last_train=[]
    for epoch in range(num_of_epochs_each_time):
        
        logger.info('>>>>>>>> Epoch %3d' % (base*num_of_epochs_each_time+epoch))
        t_top1, t_top5, t_loss = process.train_all_times(train_loader, mw,num_solution,T, criterion, epoch, monitors, args,base*num_of_epochs_each_time)
        
        prev_last=last_train
        last_train=[times,t_top1]

        this_training_list.append(t_top1)
        count+=1

        if num_of_epochs_each_time == count:
            mw.step_a()
            mw.zero_grad()
            count=0
        
    print("Current : ",this_training_list)
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
