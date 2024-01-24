import torch as t
import copy
import numpy as np


from .quantizer import Quantizer
from torch.autograd.function import InplaceFunction
from torch.autograd.variable import Variable

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
    def forward(ctx, x , x_prev):

        return x

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output ,grad_output

class Save_prev_params(InplaceFunction):

    @staticmethod
    def forward(ctx, x , a,name,xdivs,Qp,Qn):
        ctx.save_for_backward(a,xdivs)
        ctx.name=name
        ctx.Qp=Qp
        ctx.Qn=Qn

        return x

    @staticmethod
    def backward(ctx, grad_output):
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

        return None , None, None,None,None,None

class Calc_grad_a_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, x , a,name):
        ctx.save_for_backward(a)
        ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors

        a_grad = -1 * (grad_output * a) * save_gradients.get_grad(ctx.name).cuda()*save_gradients.get_mult(ctx.name).cuda() * 0.01

        #where_flip = t.where(t.sign(grad_output).ne(t.sign(save_gradients.get_grad_for_flip(ctx.name))), 1, 0)
        #new_flip_count = t.where(where_flip.eq(1), save_gradients.get_flip_count(ctx.name) + 1, 0)
        #save_gradients.update_flip_count(ctx.name,new_flip_count)

        #save_gradients.update_grad_for_flip(ctx.name,grad_output * a)
        return grad_output * a, a_grad, None

class Clamp_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, i,x_parallel, min_val, max_val,a):
        #ctx._mask1 = (i.ge(min_val/a) * i.le(max_val/a))
        maxdiva = max_val / a
        mindiva = min_val / a
        ctx._mask1 = i.ge(mindiva) * i.le(maxdiva)
        ctx._mask2 = (x_parallel.ge(min_val) * x_parallel.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):

        mask1 = Variable(ctx._mask1.type_as(grad_output.data))

        mask2 = Variable(ctx._mask2.type_as(grad_output.data))
        return grad_output * mask1,grad_output * mask2, None, None,None


class save_gradients():
    prev_grad={}
    prev_mult={}
    prev_grad_for_flip={}
    flip_count={}

    @staticmethod
    def get_mult(name):
        return save_gradients.prev_mult[name]

    @staticmethod
    def update_mult(name, mult):
        save_gradients.prev_mult.update({name: mult})

    @staticmethod
    def get_grad(name):
        return save_gradients.prev_grad[name]

    @staticmethod
    def update_grad(name,grad):
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
        count_inst += 1
        save_gradients.update_grad(self.name,t.nn.Parameter(t.zeros(1)))
        save_gradients.update_mult(self.name, t.nn.Parameter(t.zeros(1)))
        self.is_weight=False
        self.mean_of_input=0
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
        self.learn_Qn = True
        self.prev_x=t.zeros(1,device ="cuda")
        self.direction=t.zeros(1,device ="cuda")
        self.osc_counter=t.zeros(1,device ="cuda")

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            #print("check dim : ",x.size()," check size : ",x.size())
            self.is_weight=True
            self.a=t.nn.Parameter(t.ones(x.size()))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            self.a = t.nn.Parameter(t.ones(1))

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
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x

    def forward_no_quantization(self, x):#no quantization
        return x


    def forward(self, x):#with learn a gdtuo
        #learn STE
        self.mean_of_input=x.mean()
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        if self.is_weight:
            self.check_oscillations(x)

            x_parallel = x.detach()
            x_parallel = x_parallel / s_scale
            xdivs_save=x_parallel.detach()
            x_prev = Save_prev_params.apply(x, self.a, self.name, xdivs_save, self.thd_pos, self.thd_neg)
            x = Calc_grad_a_STE.apply(x,self.a,self.name)

            x = x / s_scale.detach()
            x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a)
            x = round_pass(x)
            x = x * s_scale
            x = split_grad.apply(x,x_prev)

        else:
            x = x / s_scale

            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale

        return x


