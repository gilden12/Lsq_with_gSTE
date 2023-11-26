import torch as t
import copy
import numpy as np

from .quantizer import Quantizer


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
        #print("first is: ", thd_learn, " and second: ", thd)

count_inst=0
class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        global count_inst
        self.name=count_inst
        count_inst+=1
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
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

        self.learn_Qn = True
        self.learn_thd_pos = t.nn.Parameter(t.tensor([self.thd_pos], dtype=t.float32))
        self.learn_thd_pos.retain_grad()
    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):

        if self.learn_Qn:
            if self.per_channel:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)

            w_q = LSQ.apply(x, self.s ,self.learn_thd_pos, s_grad_scale,self.thd_neg, self.thd_pos,self.name)

            return w_q
        else:
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



class LSQ(t.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha ,learn_thd, g, Qn, Qp,name):
        #assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha,learn_thd)
        ctx.other = g, Qn, Qp, name
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        with t.enable_grad():
            weight, alpha, learn_thd = ctx.saved_tensors
            g, Qn, Qp, name = ctx.other
            update_lists(learn_thd.detach(), Qp, name)
            q_w = weight.detach() / alpha

            indicate_small = (q_w < Qn).float()
            #indicate_big = (q_w < Qp).float()

            indicate_big = (q_w > learn_thd).float()

            indicate_middle = t.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big

            #indicate_middle = 1.0 - indicate_small - indicate_big # Thanks to @haolibai

            #Original code:
            #grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
            #Change:
            #grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_weight * g)
            grad_alpha = ((indicate_small * Qn + indicate_big * learn_thd + indicate_middle * (-q_w + q_w.round())) * grad_weight * g)

            grad_weight = indicate_middle * grad_weight

            grad_thd=t.autograd.grad(grad_alpha.mean(), learn_thd)[0]*grad_alpha.mean()

            return grad_weight, grad_alpha, grad_thd, None, None, None, None
