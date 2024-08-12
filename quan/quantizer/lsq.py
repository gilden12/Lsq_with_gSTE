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

thd_learn_list=[]
thd_list=[]
x=np.array([0])
def update_lists(thd_learn,thd,layerIdx):
    if layerIdx == 5:
        global x
        x=np.append(x,copy.deepcopy(thd_learn).data.cpu())
        thd_learn_list.append(copy.deepcopy(thd_learn).data.cpu())
        thd_list.append(copy.deepcopy(thd))

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

class Save_prev_params(InplaceFunction):

    @staticmethod
    def forward(ctx, x , a,name,xdivs,Qp,Qn):
        ctx.save_for_backward(a,xdivs)
        ctx.name=name
        ctx.Qp=Qp
        ctx.Qn=Qn
        ctx.sizex=x.shape
        ctx.sizea=a.shape
        #print("cehck the name ub save pre pramara ",name)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output == None:
            print (" None here")
        a,xdivs=ctx.saved_tensors
        sdivx=1/xdivs
        save_gradients.update_grad(ctx.name, grad_output)

        #a_tensor=t.full(sdivx.size(),a.item()).cuda()
        a_tensor = a

        checkge0 = xdivs.ge(0)

        mult_calc = t.where(checkge0*a_tensor.le(ctx.Qp*sdivx-0.5), 1, 0)
        mult_calc = t.where(checkge0*a_tensor.le(ctx.Qp*sdivx)*a_tensor.ge(ctx.Qp*sdivx-0.5), (1 + a), mult_calc)
        mult_calc = t.where(checkge0*a_tensor.le(ctx.Qp*sdivx+0.5)*a_tensor.ge(ctx.Qp*sdivx), a, mult_calc)

        checkle0 = xdivs.le(0)
        mult_calc = t.where(checkle0 * a_tensor.le(ctx.Qn * sdivx - 0.5), 1, mult_calc)
        mult_calc = t.where(checkle0 * a_tensor.le(ctx.Qn * sdivx) * a_tensor.ge(ctx.Qn * sdivx - 0.5), (1 + a), mult_calc)
        mult_calc = t.where(checkle0 * a_tensor.le(ctx.Qn * sdivx + 0.5) * a_tensor.ge(ctx.Qn * sdivx), a, mult_calc)

        save_gradients.update_mult(ctx.name, mult_calc)

        return t.zeros(ctx.sizex).cuda() , t.zeros(ctx.sizea).cuda(), None,None,None,None


class grad_a(InplaceFunction):

    @staticmethod
    def forward(ctx, a,name,x,a_grad):
        ctx.sizex=x.shape
        ctx.save_for_backward(a_grad)
        ctx.name=name
        return x.detach()*a

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output == None:
            print (" None here")
        a_grad, = ctx.saved_tensors
        #a_grad = save_gradients.get_grad(ctx.name).cuda()*save_gradients.get_mult(ctx.name).cuda()

        #where_flip = t.where(t.sign(grad_output).ne(t.sign(save_gradients.get_grad_for_flip(ctx.name))), 1, 0)
        #new_flip_count = t.where(where_flip.eq(1), save_gradients.get_flip_count(ctx.name) + 1, 0)
        #save_gradients.update_flip_count(ctx.name,new_flip_count)

        #save_gradients.update_grad_for_flip(ctx.name,grad_output * a)
        #print("grad_output.detach()*a_grad.detach() ",grad_output.detach()*a_grad.detach())
        #print("here 2 ")
        #return grad_output.detach()*a_grad.detach(), None,t.zeros(ctx.sizex).cuda(),None
        return t.ones(a_grad.shape).cuda(), None,t.zeros(ctx.sizex).cuda(),None


class Calc_grad_a_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, x , a,name):
        ctx.save_for_backward(a)
        ctx.name=name
        #ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #if grad_output == None:
        #    print (" None here")
        a, = ctx.saved_tensors
        #print("cehck nameeee : ",ctx.name)
        #a_grad = save_gradients.get_grad(ctx.name).cuda()*save_gradients.get_mult(ctx.name).cuda()
        #grad_out=grad_a.apply(a,ctx.name,grad_output.detach(),a_grad.detach())
        #print("here 1 ")
        #grad_out=grad_output
        #if grad_output.sum() == 0 or grad_output == None:
        #    print("grad name is : ",ctx.name," grad is : ",grad_output)
        #print("this is a : ",a)
        #print(" a shape is : ",grad_output)
        #print(" this is the entire shape : ",(a*grad_output))
        #print("is equal ",t.equal(a*grad_output,grad_output))
        #if not t.equal(a*grad_output,grad_output):
        #    print(" grad_output shape is : ",grad_output.shape)
        #    print(" this is the entire shape : ",(a*grad_output).shape)
        return a*grad_output.detach(), None, None

class debugging_cehck(InplaceFunction):

    @staticmethod
    def forward(ctx, x,name):
        ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        print("cehck name ",ctx.name)
        return grad_output,None
class Calc_grad_a_STE_meta(InplaceFunction):

    @staticmethod
    def forward(ctx, x , meta_network,name,ste_const):
        ctx.save_for_backward(x,ste_const)

        ctx.name=name
        ctx.meta_network=meta_network
        
        #ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #if grad_output == None:
        #    print (" None here")
        x,ste_const = ctx.saved_tensors
        #print("cehck nameeee : ",ctx.name)
        #a_grad = save_gradients.get_grad(ctx.name).cuda()*save_gradients.get_mult(ctx.name).cuda()
        #grad_out=grad_a.apply(a,ctx.name,grad_output.detach(),a_grad.detach())
        #print("here 1 ")
        #grad_out=grad_output
        #if grad_output.sum() == 0 or grad_output == None:
        #    print("grad name is : ",ctx.name," grad is : ",grad_output)
        flatten_weight = x.view(-1,1).detach()
        flatten_grad = grad_output.view(-1,1).detach()
        #print("chik :",grad_output.detach()+ctx.ste_const*((ctx.meta_network(flatten_weight)*flatten_grad).reshape(grad_output.detach().shape)))
        return grad_output.detach()+ste_const*((ctx.meta_network(flatten_weight)*flatten_grad).reshape(grad_output.detach().shape)), None, None, t.zeros(ste_const.size()).cuda()
    
class Calc_grad_a_STE_Meta(InplaceFunction):

    @staticmethod
    def forward(ctx,x, meta_network,name,a):
        
        ctx.save_for_backward(x,a)
        ctx.name=name
        ctx.meta_network=meta_network
        #ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #if grad_output == None:
        #    print (" None here")
        x,a = ctx.saved_tensors
        #print("cehck nameeee : ",ctx.name)
        #a_grad = save_gradients.get_grad(ctx.name).cuda()*save_gradients.get_mult(ctx.name).cuda()
        #grad_out=grad_a.apply(a,ctx.name,grad_output.detach(),a_grad.detach())
        #print("here 1 ")
        #grad_out=grad_output
        #if grad_output.sum() == 0 or grad_output == None:
        #    print("grad name is : ",ctx.name," grad is : ",grad_output)
        #if a ==None or b==None:
        #    print("ererereroreroere")
        #if grad_output == None:
        #    print("issue here now")
        #else:
        #    print("check here now")
        #print("b is : ",b.sum())
        #grad_output=t.transpose(grad_output.flatten(), 0).detach()
        #print("here done ",ctx.name)
        #for name,param in ctx.meta_network.named_parameters():
        #        print("grads of net in back : ",param.grad)
        flatten_weight = x.view(-1,1).detach()
        flatten_grad = grad_output.view(-1,1).detach()      
        return ((ctx.meta_network(flatten_weight)*flatten_grad).reshape(grad_output.detach().shape)), None, None, None
        #return a*grad_output.detach(), None, None,None

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

class Clamp_MAD(InplaceFunction):

    @staticmethod
    def forward(ctx, i,x_parallel, min_val, max_val,a):
        #ctx._mask1 = (i.ge(min_val/a) * i.le(max_val/a))
        ctx.shapea=a.shape
        maxdiva = max_val / a
        mindiva = min_val / a
        ctx._mask1 = i.ge(mindiva) * i.le(maxdiva)
        ctx._mask1_MAD = t.where(ctx._mask1,1,t.where(i.le(0),min_val/i,max_val/i))
        #print("ctx._mask1_MAD is : ",ctx._mask1_MAD)
        ctx._mask2 = (x_parallel.ge(min_val) * x_parallel.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output == None:
            print (" None here")
        mask1 = Variable(ctx._mask1_MAD.type_as(grad_output.data))

        mask2 = Variable(ctx._mask2.type_as(grad_output.data))
        return grad_output * mask1,grad_output * mask2, None, None,t.zeros(ctx.shapea).cuda()

class save_gradients():
    prev_grad={}
    prev_mult={}
    prev_grad_for_flip={}
    flip_count={}
    prev_vdivs={}
    @staticmethod
    def get_mult(name):
        return save_gradients.prev_mult[name]

    @staticmethod
    def update_mult(name, mult):
        #print(" check update mule : ",name)
        save_gradients.prev_mult.update({name: mult})

    @staticmethod
    def get_vdivs(name):
        #print("name in get_vdivs : ",name)

        return save_gradients.prev_vdivs[name]

    @staticmethod
    def update_vdivs(name, vdivs):
        #print("name in update_vdivs : ",name)

        save_gradients.prev_vdivs.update({name: vdivs.detach()})

    @staticmethod
    def get_grad(name):
        #print("name in get_grad : ",name)
        return save_gradients.prev_grad[name]

    @staticmethod
    def update_grad(name,grad):
        #print("name in update_grad : ",name)
        save_gradients.prev_grad.update({name: grad})

    @staticmethod
    def get_grad_for_flip(name):
        return save_gradients.prev_grad_for_flip[name]

    @staticmethod
    def update_grad_for_flip(name, grad):
        save_gradients.prev_grad_for_flip.update({name: grad})

    @staticmethod
    def get_flip_count(name):
        return save_gradients.flip_count[name]

    @staticmethod
    def update_flip_count(name, grad):
        save_gradients.flip_count.update({name: grad})

class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        global count_inst
        self.name = count_inst
        #print("aft self.name :",self.name )
        count_inst += 1
        save_gradients.update_grad(self.name,t.nn.Parameter(t.zeros(1)))
        save_gradients.update_mult(self.name, t.nn.Parameter(t.zeros(1)))
        self.is_weight=False
        self.mean_of_input=0
        self.get_v_hat_grads = False
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

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones(1))
        self.a = t.nn.Parameter(t.ones(1))
        self.b = t.nn.Parameter(t.zeros(1))#used for meta network linear layer
        self.v_hat = t.nn.Parameter(t.ones(1))
        self.learn_Qn = True
        self.prev_x=t.zeros(1,device ="cuda")
        self.direction=t.zeros(1,device ="cuda")
        self.osc_counter=t.zeros(1,device ="cuda")
        self.using_gdtuo=False
        self.num_solution=0
        self.T=0
        self.counter=0
        self.meta_network = MetaFC(100,False)
        self.meta_modules_STE_const = t.nn.Parameter(t.zeros(1))
        self.x_hat=t.nn.Parameter(t.ones(1))
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
        print("here now")
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            #print("check dim : ",x.size()," check size : ",x.size())
            self.is_weight=True
            self.a=t.nn.Parameter(t.ones(x.size()))
            self.x_hat=t.nn.Parameter(t.ones(x.size()))
            self.v_hat = t.nn.Parameter(t.ones(x.size()))
            self.num_share_params=1
            if self.num_solution == 2 or self.num_solution==8 or self.num_solution==9 or self.num_solution==10 or self.num_solution==11:
                
                if self.a_per == 0:#a per element
                    my_list = [i for i in x.size()]
                if self.a_per == 1:#a per layer
                    my_list = [1]
                if self.a_per == 2:#a per channel
                    my_list = [i for i in x.size()]
                    my_list = [my_list[0]] + [1] * (len(my_list) - 1)

                my_list=[math.ceil(self.T/self.num_share_params)]+my_list
                my_tupel=tuple(my_list)
                self.a=t.nn.Parameter(t.ones(my_tupel,dtype=t.float16))
                print("dimenstions of x ",self.a.shape)
                print("check for me ",[i for i in x.size()])
                
            if self.num_solution == 7:
                meta_copy=copy.deepcopy(self.meta_network)
                
                a_list = t.nn.ModuleList()
                for i in range(self.T):
                    a_list.append(meta_copy)
                    meta_copy=copy.deepcopy(self.meta_network)
                    
                self.meta_modules = a_list

                self.meta_modules_STE_const = t.nn.Parameter(t.zeros(self.T))
                # ste_const=copy.deepcopy(self.meta_modules_STE_const)
                # ste_const_list = []
                # for i in range(self.T):
                #     ste_const_list.append(ste_const)
                #     ste_const=copy.deepcopy(self.meta_modules_STE_const)
                
                # self.meta_modules_STE_const_list=ste_const_list

            if self.num_solution == 1.5:
                self.b = t.nn.Parameter(t.zeros(x.size()))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            self.a = t.nn.Parameter(t.ones(1))
            self.v_hat = t.nn.Parameter(t.ones(1))

    def check_oscillations(self,x):
        if not self.prev_x.shape==t.zeros(1,device ="cuda").shape:
            #print((x/self.s).round())
            new_direction=t.where(self.prev_x.round().lt((x/self.s).round()), 1, self.direction)
            new_direction = t.where(self.prev_x.round().gt((x/self.s).round()), 2, new_direction)
            osc_counter_new = t.where(new_direction.ne(self.direction), self.osc_counter + 1,self.osc_counter)
            #if self.name ==1 and not osc_counter_new.sum() == self.osc_counter.sum():
                #print("sizes : ", self.new_direction.size(), self.direction.size(), self.osc_counter.size())
                #print("osc counter = ",self.osc_counter)
                #print(" sumes : ",osc_counter_new.sum(),self.osc_counter.sum())


            self.osc_counter=osc_counter_new
            #if t.where(a)
            self.direction=new_direction
        #print(self.prev_x)
        #if self.name==3:
        #    print(" Num osc : ",self.osc_counter.sum())
        self.prev_x=(x/self.s).detach()


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

    
    def forward_analytical_gSTE(self, x):#with learn a gdtuo generalized optimizer
        #learn STE
        self.mean_of_input=x.mean()
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        if self.is_weight:
            #self.check_oscillations(x)
            #print("check str name",self.strname)
            
            
            if self.get_v_hat_grads==False:
                #print("In bad")

                x_parallel = x.detach()
                x_parallel = x_parallel / s_scale
                xdivs_save=x_parallel.detach()
                x_prev=x.detach()
                x = Calc_grad_a_STE.apply(x,self.a,self.name)
                x_prev = Save_prev_params.apply(x, self.a, self.strname, xdivs_save, self.thd_pos, self.thd_neg)#changed the position to be after calc_grad
                x = x / s_scale.detach()
                x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a)
                x = round_pass(x)
                x = x * s_scale
                
            else:
                with t.no_grad():
                    x_prev=x.detach()
                    s_scale = grad_scale(self.s, s_grad_scale)
                    x = x / s_scale
                    vdivs = x.detach()
                    vdivs_Qq = t.where(vdivs>0 , vdivs*self.thd_pos , vdivs*self.thd_neg)
                    save_gradients.update_vdivs(self.strname,vdivs_Qq)
                    x = t.clamp(x, self.thd_neg, self.thd_pos)
                    x = round_pass(x)
                    x = x * s_scale
            x = split_grad.apply(x,x_prev,self.v_hat)        

        # else:
        #     #print(" check if here : ",self.strname)
        #     #print("str name in elses : ",self.strname)
        #     x = x / s_scale

        #     x = t.clamp(x, self.thd_neg, self.thd_pos)
        #     x = round_pass(x)
        #     x = x * s_scale

        return x
    
    def forward_delayed_updates(self, x):#original forward from lsq
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
            x = Calc_grad_a_STE.apply(x,self.a,self.name)
            #x = Calc_grad_a_STE_Meta.apply(x,self.meta_network,self.name,self.a)
            #x_prev = Save_prev_params.apply(x, self.a, self.strname, xdivs_save, self.thd_pos, self.thd_neg)#changed the position to be after calc_grad
            x = x / s_scale.detach()
            x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a)
            x = round_pass(x)
            x = x * s_scale
            x = split_grad.apply(x,x_prev,self.v_hat)        
        else:
            x = x / s_scale

            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale

        return x
    

    def forward_delayed_updates_meta_quant(self, x):#original forward from lsq
            #for name,param in self.meta_network.named_parameters():
            #    print("grads of net before forward : ",param.grad)
            #print("check s grad : ",self.s.grad )
            #print("check x grad : ", x.grad )
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
                
                x = Calc_grad_a_STE_Meta.apply(x,self.meta_network,self.name,self.a)
                x = x / s_scale
                x = t.clamp(x, self.thd_neg, self.thd_pos)
                x = round_pass(x)
                x = x * s_scale
                """x_parallel = x.detach()
                x_parallel = x_parallel / s_scale
                xdivs_save=x_parallel.detach()
                x_prev=x.detach()
                x = Calc_grad_a_STE_Meta.apply(x,self.a,self.b,self.name)
                x = x / s_scale.detach()
                x = t.clamp(x, self.thd_neg, self.thd_pos)
                x = round_pass(x)
                x = x * s_scale
                x = split_grad.apply(x,x_prev,self.v_hat)        """
            else:
                x = x / s_scale

                x = t.clamp(x, self.thd_neg, self.thd_pos)
                x = round_pass(x)
                x = x * s_scale

            return x

    def forward_all_times_for_MAD(self, x):#original forward from lsq
        #print("thd neg and thd pos are :",self.thd_neg,self.thd_pos)
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

            x = Calc_grad_a_STE.apply(x,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))],self.name)

            x = x / s_scale.detach()

            x = Clamp_MAD.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))])
            x = round_pass(x)
            x = x * s_scale
            x = split_grad.apply(x,x_prev,self.v_hat)
            
            self.counter+=1

        return x


    def forward_all_times(self, x):#original forward from lsq
        #print("thd neg and thd pos are :",self.thd_neg,self.thd_pos)
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
            #print("check a value : ",self.a)
            #print("self.counter is : ",self.counter)
            #print("The shape for weight x is : ", x.shape)
            x_parallel = x.detach()
            x_parallel = x_parallel / s_scale
            xdivs_save=x_parallel.detach()
            x_prev=x.detach()
            use_ste_end=False
            if self.num_solution == 7:
                #print(" in forward, count = ",self.counter," count%T = ",self.counter%self.T)
                #print("cehck 4545 ",self.meta_modules_STE_const[self.counter%(self.T)])
                
                x = Calc_grad_a_STE_meta.apply(x,self.meta_modules[math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params)],self.name,self.meta_modules_STE_const[self.counter%(self.T)])
            else:
                #print("check countres: ",math.floor(self.counter/self.num_share_params)," check s : ",(self.T/self.num_share_params))
                #print("check if eq 1 : ",t.all(self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))] == 1))
                #print("check if eq 1 : ",(self.T/self.num_share_params))
                
                #print("a num : ",int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params)))
                if self.set_use_last_a_trained:
                    x = Calc_grad_a_STE.apply(x,self.a[-2],self.name)
                else:
                    if int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params)) == int((self.T/self.num_share_params)-1) and use_ste_end==False:
                        #print("Im here : ",int(math.floor(self.counter/self.num_share_params-1)%(self.T/self.num_share_params)))
                        x = Calc_grad_a_STE.apply(x,self.a[int(math.floor(self.counter/self.num_share_params-1)%(self.T/self.num_share_params))],self.name)
                    else:
                        x = Calc_grad_a_STE.apply(x,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))],self.name)
            #if self.counter%self.T == 0:
            #    print("Now equals zero !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #if (self.counter%self.T)%10== 0 or (self.counter%self.T)==self.T-2 or (self.counter%self.T)==self.T-1 or (self.counter%self.T)==self.T-3:
            #    print("check self.counterelf.T :",self.counter%self.T)
            #    if self.a.grad != None:
            #        print((self.a.grad[-1]).sum())
            #    else:
            #        print("None self.a[100].grad")

            #if self.a.grad != None:
            #    is_all_ones = t.all((self.a[-2]) == 1).item()
            #    print("check if all ones : ",is_all_ones)  # Output: True

            #print("self.a.size() : ",self.a.size())
            #print("self.a[self.counter%self.T].size() : ",self.a[self.counter%self.T].size())
            #print("check self a : ",self.a[1].sum())
            #x_prev = Save_prev_params.apply(x, 1.0, self.strname, xdivs_save, self.thd_pos, self.thd_neg)#changed the position to be after calc_grad
            #x_prev = Save_prev_params.apply(x, self.a, self.strname, xdivs_save, self.thd_pos, self.thd_neg)#changed the position to be after calc_grad
            x = x / s_scale.detach()
            if self.num_solution == 7:
                x = t.clamp(x, self.thd_neg, self.thd_pos)
            else:
                x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))])
            #x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a)
            x = round_pass(x)
            x = x * s_scale
            x = split_grad.apply(x,x_prev,self.v_hat)
            
            self.counter+=1
            
        #else:
        #    print("The shape for activation x is : ", x.shape)
        #    x_original=x
        #    x = x / s_scale

        #    x = t.clamp(x, self.thd_neg, self.thd_pos)
        #    x = round_pass(x)
        #    x = x * s_scale

        return x
    

    def forward_baseline_no_quant(self, x):#original forward from lsq
        #print("x.detach() ",self.x_hat)
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
            print("ERORRRRRRR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
                x_original=x
                # x = x / s_scale

                # x = t.clamp(x, self.thd_neg, self.thd_pos)
                # x = round_pass(x)
                # x = x * s_scale
                return x_original

        

    def forward(self, x):
        
        #return self.forward_no_quantization(x)

        if self.num_solution == -1:
            return self.forward_original( x)
        
        elif self.num_solution == 0 or self.num_solution == 5:
            return self.forward_analytical_gSTE( x)
        elif self.num_solution == 1:
            return self.forward_delayed_updates( x)
        elif self.num_solution == 1.5 or self.num_solution == 6:
            return self.forward_delayed_updates_meta_quant(x)
        elif self.num_solution == 2 or self.num_solution == 7 or self.num_solution == 10 or self.num_solution == 11:
            return self.forward_all_times( x)
        elif self.num_solution == 8:
            return self.forward_baseline_no_quant( x)
        elif self.num_solution == 9:
            return self.forward_all_times_for_MAD( x)
        
        else:
            print("Solution not defined")
        


class simple_grad(InplaceFunction):

    @staticmethod
    def forward(ctx, x ):

        return x.detach()

    @staticmethod
    def backward(ctx, grad_output):
        
        return grad_output.detach()

class MetaMultiFC(nn.Module):

    def __init__(self, hidden_size = 10, use_nonlinear=None):
        super(MetaMultiFC, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=hidden_size, bias=False)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        self.use_nonlinear = use_nonlinear

    def forward(self, x):
        x = self.linear1(x)
        if self.use_nonlinear == 'relu':
            x = F.relu(x)
        elif self.use_nonlinear == 'tanh':
            x = t.tanh(x)
        # x = self.linear2(x)
        # if self.use_nonlinear == 'relu':
        #     x = F.relu(x)
        # elif self.use_nonlinear == 'tanh':
        #     x = t.tanh(x)
        x = self.linear3(x).detach()



        return x
    
class MetaFC(nn.Module):

    def __init__(self, hidden_size = 1500, symmetric_init=False, use_nonlinear=None):
        super(MetaFC, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=hidden_size, bias=False)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        if symmetric_init:
            self.linear1.weight.data.fill_(1.0 / hidden_size)
            self.linear2.weight.data.fill_(1.0)

        self.use_nonlinear = use_nonlinear

    def forward(self, x):

        x = self.linear1(x)
        if self.use_nonlinear is 'relu':
            x = F.relu(x)
        elif self.use_nonlinear is 'tanh':
            x = t.tanh(x)
        x = self.linear2(x)

        return x

