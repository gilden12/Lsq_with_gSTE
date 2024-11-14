import torch as t
import copy
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math

from .quantizer import Quantizer
from torch.autograd.function import InplaceFunction
from torch.autograd.variable import Variable
#from ...main import *
import sys
#sys.path.append('/home/gild/Lsq_with_gSTE')
#import .main


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


count_inst=0


class split_grad(InplaceFunction):

    @staticmethod
    def forward(ctx, x , x_prev,v_hat):

        return x

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output == None:
            print (" None here")
        return grad_output ,grad_output,grad_output



class Calc_grad_a_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, x , a,name):
        ctx.save_for_backward(a)
        ctx.name=name
        #ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return a*grad_output.detach(), None, None


class Calc_grad_a_STE_dupl(InplaceFunction):

    @staticmethod
    def forward(ctx, x ,x_hat, a,name):
        ctx.save_for_backward(a)
        ctx.name=name
        #ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return a*grad_output.detach(),a*grad_output, None, None


class Clamp_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, i,x_parallel, min_val, max_val,a):
        #ctx._mask1 = (i.ge(min_val/a) * i.le(max_val/a))
        ctx.shapea=a.shape
        maxdiva = max_val / a
        mindiva = min_val / a
        ctx._mask1 = i.ge(mindiva) * i.le(maxdiva)
        ctx._mask2 = (x_parallel.ge(min_val) * x_parallel.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output == None:
            print (" None here")
        mask1 = Variable(ctx._mask1.type_as(grad_output.data))

        mask2 = Variable(ctx._mask2.type_as(grad_output.data))
        return grad_output * mask1,grad_output * mask2, None, None,t.zeros(ctx.shapea).cuda()


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        global count_inst
        self.name = count_inst
        count_inst += 1
        self.is_weight=False
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            #self.is_weight =True
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        isBinary=False
        if isBinary:
            self.thd_neg = 0
            self.thd_pos = 1

        self.per_channel = per_channel
        self.x_hat=t.nn.Parameter(t.ones(1))

        self.s = t.nn.Parameter(t.ones(1))
        self.a = t.nn.Parameter(t.ones(1))
        self.v_hat = t.nn.Parameter(t.ones(1))
        self.num_solution=0
        self.T=0
        self.counter=0
        self.meta_modules_STE_const = t.nn.Parameter(t.zeros(1))
        self.set_use_last_a_trained=False
    def update_strname(self,strname):
        self.strname=strname
    def use_last_a_trained(self):
        self.set_use_last_a_trained=True
    def update_list_for_lsq(self,num_solution,list_for_lsq):
        self.num_solution=num_solution
        self.T=list_for_lsq[0]
        self.a_per=list_for_lsq[1]
        self.num_share_params=list_for_lsq[2]

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            #print("check dim : ",x.size()," check size : ",x.size())
            self.is_weight=True
            self.a=t.nn.Parameter(t.ones(x.size()))
            self.x_hat=t.nn.Parameter(t.ones(x.size()))
            self.v_hat = t.nn.Parameter(t.ones(x.size()))
            self.num_share_params=1
            if self.num_solution == 2 or self.num_solution==8 or self.num_solution==9 or self.num_solution==10 or self.num_solution==11 or self.num_solution == 12:
                
                if self.a_per == 0:#a per element
                    my_list = [i for i in x.size()]
                if self.a_per == 1:#a per layer
                    my_list = [1]
                if self.a_per == 2:#a per channel
                    my_list = [i for i in x.size()]
                    my_list = [my_list[0]] + [1] * (len(my_list) - 1)

                my_list=[math.ceil(self.T/self.num_share_params)]+my_list
                my_tupel=tuple(my_list)
                #self.a=t.nn.Parameter(t.ones(my_tupel,dtype=t.float16))
                self.a=t.nn.Parameter(t.ones(my_tupel))
                print("shape self.a : ",self.a.shape,"shape self.x_hat : ",self.x_hat.shape)
            if self.num_solution == 7:
                meta_copy=copy.deepcopy(self.meta_network)
                
                a_list = t.nn.ModuleList()
                for i in range(self.T):
                    a_list.append(meta_copy)
                    meta_copy=copy.deepcopy(self.meta_network)
                    
                self.meta_modules = a_list

                self.meta_modules_STE_const = t.nn.Parameter(t.zeros(self.T))

        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            self.a = t.nn.Parameter(t.ones(1))
            self.v_hat = t.nn.Parameter(t.ones(1))

    def forward_original(self, x):#original forward from lsq
        if self.per_channel:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        else:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        s_scale = grad_scale(self.s, s_grad_scale)
        if self.is_weight:
            x = x / s_scale
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale
        return x

    def forward_no_quantization(self, x):#no quantization
        return x


    def forward_all_times(self, x):#original forward from lsq
        if self.per_channel:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        else:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0

        s_scale = grad_scale(self.s, s_grad_scale)
        
        if self.is_weight:
            x_parallel = x.detach()
            x_parallel = x_parallel / s_scale
            xdivs_save=x_parallel.detach()
            x_prev=x.detach()
            use_ste_end=False
            
            if self.set_use_last_a_trained:
                x = Calc_grad_a_STE.apply(x,self.a[-2],self.name)
            else:
                if int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params)) == int((self.T/self.num_share_params)-1) and use_ste_end==False:
                    x = Calc_grad_a_STE.apply(x,self.a[int(math.floor(self.counter/self.num_share_params-1)%(self.T/self.num_share_params))],self.name)
                else:
                    x = Calc_grad_a_STE.apply(x,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))],self.name)
            

            x = x / s_scale.detach()
            if self.num_solution == 7:
                x = t.clamp(x, self.thd_neg, self.thd_pos)
            else:
                x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))])
            x = round_pass(x)

            #print("bef",x)
            x = x * s_scale
            x = split_grad.apply(x,x_prev,self.v_hat)
            
            self.counter+=1
            #print("aft",x)

        return x
    
    def forward_less_greedy_Updates(self, x):#original forward from lsq
        if self.per_channel:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        else:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0

        s_scale = grad_scale(self.s, s_grad_scale)
        
        if self.is_weight:
            x_parallel = x.detach()
            x_parallel = x_parallel / s_scale
            xdivs_save=x_parallel.detach()
            x_prev=x.detach()
            use_ste_end=True
            
            if self.set_use_last_a_trained:
                x = Calc_grad_a_STE_dupl.apply(x,self.a[-2],self.name)
            else:
                if int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params)) == int((self.T/self.num_share_params)-1) and use_ste_end==False:
                    x = Calc_grad_a_STE_dupl.apply(x,self.x_hat,self.a[int(math.floor(self.counter/self.num_share_params-1)%(self.T/self.num_share_params))],self.name)
                else:
                    x = Calc_grad_a_STE_dupl.apply(x,self.x_hat,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))],self.name)
            

            x = x / s_scale.detach()
            if self.num_solution == 7:
                x = t.clamp(x, self.thd_neg, self.thd_pos)
            else:
                x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))])
            x = round_pass(x)

            #print("bef",x)
            x = x * s_scale
            x = split_grad.apply(x,x_prev,self.v_hat)
            
            self.counter+=1
            #print("aft",x)

        return x
    

    def forward_baseline_no_quant(self, x):#original forward from lsq
        if self.per_channel:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        else:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        s_scale = grad_scale(self.s, s_grad_scale)
        if t.equal(t.ones(x.size()).to(device='cuda'),self.x_hat):
            x_original=x
            x = x / s_scale
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            
            x = x * s_scale
            self.x_hat.data =(x.detach())
            if self.is_weight:
                return self.x_hat
            else:
                return x_original
        else:
            if self.is_weight:
                return self.x_hat
            else:
                return x

        

    def forward(self, x):
        
        if self.num_solution == -1:
            return self.forward_original( x)
        
        #elif self.num_solution == 0 or self.num_solution == 5:
        #    return self.forward_analytical_gSTE( x)
        #elif self.num_solution == 1:
        #    return self.forward_delayed_updates( x)
        #elif self.num_solution == 1.5 or self.num_solution == 6:
        #    return self.forward_delayed_updates_meta_quant(x)
        elif self.num_solution == 2 or self.num_solution == 7 or self.num_solution == 10 or self.num_solution == 11:
            #print("herererer")
            return self.forward_all_times( x)
        elif self.num_solution == 8:
            return self.forward_baseline_no_quant( x)
        #elif self.num_solution == 9:
        #    return self.forward_all_times_for_MAD( x)
        elif self.num_solution == 12:
            return self.forward_less_greedy_Updates( x)
        else:
            print("Solution not defined")
        




    