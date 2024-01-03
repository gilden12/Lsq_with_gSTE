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

        a_tensor=t.full(sdivx.size(),a.item()).cuda()

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
        a_grad = -1 * (grad_output * a) * t.sum(save_gradients.get_grad(ctx.name)*save_gradients.get_mult(ctx.name), dim=0) * 0.01

        return grad_output * a, a_grad, None

class Clamp_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, i,x_parallel, min_val, max_val,a):
        ctx._mask1 = (i.ge(min_val/a) * i.le(max_val/a))
        ctx._mask2 = (x_parallel.ge(min_val) * x_parallel.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        mask1 = Variable(ctx._mask1.type_as(grad_output.data))
        mask2 = Variable(ctx._mask2.type_as(grad_output.data))
        return grad_output * mask1,grad_output * mask2, None, None,None


class Calc_grad_a(InplaceFunction):

    @staticmethod
    def forward(ctx, s,vdivs, Qp, Qn, a,name):
        ctx.save_for_backward(a)
        ctx.Qp=Qp
        ctx.Qn=Qn
        ctx.vdivs=vdivs
        ctx.name=name
        return s

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        a_grad=-1*grad_output*save_gradients.get_grad(ctx.name)*0.01
        a_grad=t.clamp(a_grad,-0.01,0.01)
        return grad_output, None, None, None, a_grad,None


class Mult_Back_a(InplaceFunction):

    @staticmethod
    def forward(ctx, x,s_scale,x_parallel, Qp, Qn, a,x_check,x_parallel2):
        ctx.save_for_backward(x,s_scale,x_parallel,a,x_check,x_parallel2)
        ctx.Qp = Qp
        ctx.Qn = Qn

        return x

    @staticmethod
    def backward(ctx, grad_output):
        x,s_scale,x_parallel,a,x_check,x_parallel2 = ctx.saved_tensors
        out=t.where(((x.eq(ctx.Qn*s_scale)+x.eq(ctx.Qp*s_scale))==1),(grad_output),(grad_output*a))
        out_parallel=t.where(((x.eq(ctx.Qn*s_scale)+x.eq(ctx.Qp*s_scale))==1),(0),(grad_output))

        return out, None, out_parallel, None, None, None,None,grad_output

class save_gradients():
    prev_grad={}
    prev_mult={}

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

class save_grad(InplaceFunction):

    @staticmethod
    def forward(ctx, s_parallel,name):
        ctx.name=name
        return s_parallel

    @staticmethod
    def backward(ctx, grad_output):
        save_gradients.update_grad(ctx.name, grad_output)
        return grad_output,None


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        global count_inst
        self.name = count_inst
        count_inst += 1
        save_gradients.update_grad(self.name,t.nn.Parameter(t.ones(1)))
        save_gradients.update_mult(self.name, t.nn.Parameter(t.ones(1)))
        self.is_weight=False
        self.mean_of_input=0
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            self.is_weight =True
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
        self.learn_thd_pos = t.nn.Parameter(t.tensor([self.thd_pos], dtype=t.float32))
        self.learn_thd_pos.retain_grad()

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))


    def forward_learn_a_for_s(self, x):#with learn a gdtuo
        #learn update for s

        #update_lists(self.a.detach(), 1, self.name)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)

        s_scale = grad_scale(self.s, s_grad_scale)
        s_parallel = s_scale.detach().requires_grad_(True)
        s_parallel2 = s_scale.detach()


        s_parallel = save_grad.apply(s_parallel,self.name)
        s_scale = Calc_grad_a.apply(s_scale, x/s_scale, self.thd_pos, self.thd_neg, self.a,self.name)

        x_parallel=x.detach()
        x_parallel2=x

        x = x.detach() / s_scale
        x_parallel=(x_parallel / s_parallel)
        x_parallel2=x_parallel2 / s_parallel2

        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x_parallel=t.clamp(x_parallel, self.thd_neg, self.thd_pos)
        x_parallel2 = t.clamp(x_parallel2, self.thd_neg, self.thd_pos)

        x = round_pass(x)
        x_parallel=round_pass(x_parallel)
        x_parallel2=round_pass(x_parallel2)


        x=x*s_scale
        x_parallel =x_parallel*s_parallel
        x_parallel2 =x_parallel2*s_parallel2


        x = Mult_Back_a.apply(x,s_scale,x_parallel, self.thd_pos, self.thd_neg, self.a,x_parallel,x_parallel2)

        return x

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


