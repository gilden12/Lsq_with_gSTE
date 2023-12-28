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
        #print("first is: ", thd_learn, " and second: ", thd)

count_inst=0

class Custom_back_new(InplaceFunction):

    @staticmethod
    def forward(ctx, s,vdivs, Qp, Qn, a):
        #ctx._mask = s.eq(Qn)+s.eq(Qp)
        ctx.save_for_backward(a,s)
        ctx.Qp=Qp
        ctx.Qn=Qn
        return s

    @staticmethod
    def backward(ctx, grad_output):
        a,s = ctx.saved_tensors
        #grad_clone=grad_output.clone()
        #print("this is a : ",a)
        mask_pass= t.logical_not(grad_output.eq(ctx.Qn) + grad_output.eq(ctx.Qp))
        with t.enable_grad():
            #print(" a is: ",a)
            #grad_input = mask_pass * grad_output * a[0] +mask_pass * grad_output- grad_output
            grad_input=mask_pass * grad_output * a+mask_pass * grad_output- grad_output
            a_grad_temp = t.autograd.grad(grad_input.mean(), a)[0]
            a_grad=a_grad_temp* grad_input.mean()
            jac = t.autograd.functional.jvp(grad_input, s, t.ones_like(s))[1]
            jac_inv = t.pinverse(jac)
            ds_dL = jac_inv

            #inv = t.autograd.grad(grad_input, s, grad_outputs=-1 * t.ones_like(grad_input),allow_unused=True)[0]
            #print(" this is inv : ",inv)
            #print("grads are: ",a_grad )
            #if grad =qpqn pass normall
            #mask_pass=Variable(ctx._mask.type_as(grad_clone.data))
            #print(" mask is : ",mask_pass)
            #print(" a is : ", a[0])
            #grad_input=grad_clone

            #grad_input=t.where(mask_pass==1,grad_output,grad_output)
            #print(" grad_input : ", grad_input.mean()," a_grad_temp : ", a_grad_temp)

            #a_grad=t.autograd.grad(grad_input.mean(),a[0])[0]*grad_input.mean()
            #print("this is grad input : ",grad_input)
            #print("this is grad output : ", grad_input)
            return grad_input, None, None, None, a_grad
        #if mask_pass
        #print(" check how : ",mask_pass)
        #print("mask shape ",mask_pass.shape," grad_clone shape ",grad_clone.shape)

class Custom_back(InplaceFunction):

    @staticmethod
    def forward(ctx, s,vdivs, Qp, Qn, a):
        ctx.save_for_backward(a)
        ctx.Qp=Qp
        ctx.Qn=Qn
        return s

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors
        mask_pass= t.logical_not(grad_output.eq(ctx.Qn) + grad_output.eq(ctx.Qp))

        with t.enable_grad():
            grad_input=mask_pass * grad_output * a[0]+mask_pass * grad_output- grad_output
            a_grad=t.autograd.grad(grad_input.mean(), a[0])[0]* grad_input.mean()
            return grad_input, None, None, None, a_grad


class save_gradients():
    prev_grad={}

    @staticmethod
    def get_grad(name):
        return save_gradients.prev_grad[name]

    @staticmethod
    def update_grad(name,grad):
        save_gradients.prev_grad.update({name: grad})

class Custom_back_gdtuo(InplaceFunction):

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
        a = ctx.saved_tensors

        mask_pass= t.logical_not(grad_output.eq(ctx.Qn) + grad_output.eq(ctx.Qp))
        #print(grad_output)
        if not isinstance(save_gradients.get_grad(ctx.name),int):
            #print("check this")
            #if (ctx.Qn in grad_output) or (ctx.Qp in grad_output):
            #    print("check1")
            #if (ctx.Qn in  save_gradients.get_grad(ctx.name)) or (ctx.Qp in  save_gradients.get_grad(ctx.name)):
            #    print("check2")
            vd=ctx.vdivs
            temp1=vd.gt(ctx.Qp)
            temp2 = vd.lt(ctx.Qn)
            intchek=t.where(temp2==1,grad_output,111)

            #if 1 in temp1 or 1 in temp2:
            #    print("check6 Qn is :",ctx.Qn," Qp is : ",ctx.Qp," intchek is : ",intchek)
            vd = grad_output
            temp1 = vd.le(0)
            temp2 = vd.ge(ctx.Qp)
            intchek=t.where(temp1==1,grad_output,111)
            #print("uhuhuh : ",temp1)
            #if 1 in temp1 or 1 in temp2:
            #    print("check4 Qn is :",ctx.Qn," Qp is : ",ctx.Qp," grad_output is : ",intchek)
        grad_input=mask_pass * grad_output * a[0]+mask_pass * grad_output- grad_output
        a_grad=-1*grad_input*save_gradients.get_grad(ctx.name)*0.01
        #a_grad=None
        #print(" a grad is - ",a_grad)
        save_gradients.update_grad(ctx.name,grad_output)
        return grad_input, None, None, None, a_grad,None


class Mult_Clamp(InplaceFunction):

    @staticmethod
    def forward(ctx, i, min_val, max_val, a):
        ctx._mask = (i.ge(min_val) * i.le(max_val))
        ctx.save_for_backward(a)
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors
        mask = Variable(ctx._mask.type_as(grad_output.data))
        #print("checa : ",a)
        grad_clone=grad_output.clone()
        mask_new = Variable(ctx._mask.type_as(grad_clone.data))

        with t.enable_grad():
            temp=grad_output.mean()* a[0]
            grad_out=temp * mask
            temp2=t.autograd.grad(grad_out.mean(), a[0])[0]
            grad_thd = temp2*grad_out.mean()*1e8
            if not t.count_nonzero(mask)==mask.sum():
                print(" mask is: ",mask)
            #print("grad_out here : ", grad_output.mean()," temp2 here : ",temp2," but the mask is: ",mask)
        return grad_clone*mask_new, None, None,grad_thd

class Inspect_layer(InplaceFunction):

    def forward(ctx, s, vdivs, Qp, Qn, a, name):
        ctx.save_for_backward(s)
        ctx.Qp = Qp
        ctx.Qn = Qn
        ctx.vdivs = vdivs
        ctx.name = name
        return s

    @staticmethod
    def backward(ctx, grad_output):
        vdivsog, = ctx.saved_tensors
        # print(grad_output)
        if not isinstance(save_gradients.get_grad(ctx.name), int):
            # print("check this")
            # if (ctx.Qn in grad_output) or (ctx.Qp in grad_output):
            #    print("check1")
            # if (ctx.Qn in  save_gradients.get_grad(ctx.name)) or (ctx.Qp in  save_gradients.get_grad(ctx.name)):
            #    print("check2")
            vd = vdivsog
            temp1 = vd.gt(ctx.Qp)
            temp2 = vd.lt(ctx.Qn)
            intchek = t.where(temp2 == 1, grad_output, 111)
            checkprob=t.where(intchek.eq(111)==1,1, vd)

            if 1 in temp1:
                print("check3 Qn is :", ctx.Qn, " Qp is : ", ctx.Qp, " intchek is : ", intchek," check prob : ",checkprob)
            vd = grad_output
            temp1 = vd.le(0)
            temp2 = vd.ge(ctx.Qp)
            intchek = t.where(temp1 == 1, grad_output, 111)
            # print("uhuhuh : ",temp1)
            # if 1 in temp1 or 1 in temp2:
            #    print("check4 Qn is :",ctx.Qn," Qp is : ",ctx.Qp," grad_output is : ",intchek)
        #print("grad_out for inspect is: ",grad_output.lt(ctx.Qn).sum()+grad_output.gt(ctx.Qp).sum())
        #if grad_output.lt(ctx.Qn).sum() !=0:
        #    print(" check it : ",grad_output," check qn : ",ctx.Qn)
        return grad_output, None, None, None, None,None

class Inspect_layer2(InplaceFunction):

    def forward(ctx, s, vdivs, Qp, Qn, a, name):
        ctx.save_for_backward(s)
        ctx.Qp = Qp
        ctx.Qn = Qn
        ctx.vdivs = vdivs
        ctx.name = name

        return s

    @staticmethod
    def backward(ctx, grad_output):
        vdivsog, = ctx.saved_tensors
        # print(grad_output)
        print(" my size for : ", vdivsog.size()," my size bak : ", grad_output.size()," x/s size ",ctx.vdivs.size())
        if not isinstance(save_gradients.get_grad(ctx.name), int):
            # print("check this")
            # if (ctx.Qn in grad_output) or (ctx.Qp in grad_output):
            #    print("check1")
            # if (ctx.Qn in  save_gradients.get_grad(ctx.name)) or (ctx.Qp in  save_gradients.get_grad(ctx.name)):
            #    print("check2")
            vd = ctx.vdivs
            temp1 = vd.gt(ctx.Qp)
            temp2 = vd.lt(ctx.Qn)
            intchek = t.where(temp2 == 1, grad_output, 111)
            checkprob=t.where(intchek.eq(111)==1,1, vd)

            #if 1 in temp1:
            #    print("check3 Qn is :", ctx.Qn, " Qp is : ", ctx.Qp, " intchek is : ", intchek," check prob : ",checkprob)
            vd = grad_output
            temp1 = vd.le(0)
            temp2 = vd.ge(ctx.Qp)
            intchek = t.where(temp1 == 1, grad_output, 111)
            # print("uhuhuh : ",temp1)
            # if 1 in temp1 or 1 in temp2:
            #    print("check4 Qn is :",ctx.Qn," Qp is : ",ctx.Qp," grad_output is : ",intchek)
        #print("grad_out for inspect is: ",grad_output.lt(ctx.Qn).sum()+grad_output.gt(ctx.Qp).sum())
        #if grad_output.lt(ctx.Qn).sum() !=0:
        #    print(" check it : ",grad_output," check qn : ",ctx.Qn)
        return grad_output, None, None, None, None,None


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        global count_inst
        self.name = count_inst
        count_inst += 1
        save_gradients.update_grad(self.name,1)
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

    def forward_new(self, x):

        if self.learn_Qn:
            if self.per_channel:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)

            w_q = LSQ.apply(x, self.s, self.a, s_grad_scale, self.thd_neg, self.thd_pos, self.name)

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
    def forward_v3(self, x):#with learn a v1
        #print("a val:", self.a)
        update_lists(self.a.detach(), 1, self.name)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)
        x = x / s_scale
        x = Mult_Clamp.apply(x, self.thd_neg, self.thd_pos, self.a)
        x = round_pass(x)
        x = x * s_scale
        #output = funcgrad.apply(x, self.a)
        return x

    def forward(self, x):#with learn a gdtuo
        #print("s val:", self.s)
        #print("a val:", self.a)
        update_lists(self.a.detach(), 1, self.name)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)
        ten1=(x / s_scale).detach()
        s_scale = Custom_back_gdtuo.apply(s_scale, ten1, self.thd_pos, self.thd_neg, self.a,self.name)
        s_scale_temp = Inspect_layer2.apply(s_scale, ten1, self.thd_pos, self.thd_neg, self.a,self.name)
        #print(" is equal : ", s_scale.size(), " sec : ", x.size())
        x = x / s_scale_temp
        #print(" new size : ", x.size())
        #x=Inspect_layer.apply(x, (x / s_scale).detach(), self.thd_pos, self.thd_neg, self.a,self.name)
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


    def forward_original(self, x):#original forward
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
    def forward(ctx, weight, alpha ,a, g, Qn, Qp,name):
        #assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha,a)
        ctx.other = g, Qn, Qp, name
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        with t.enable_grad():
            weight, alpha, a = ctx.saved_tensors
            g, Qn, Qp, name = ctx.other
            update_lists(a.detach(), 1, name)
            q_w = weight.detach() / alpha

            indicate_small = (q_w < Qn).float()
            #indicate_big = (q_w < Qp).float()

            indicate_big = (q_w > Qp).float()

            indicate_middle = t.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big

            #indicate_middle = 1.0 - indicate_small - indicate_big # Thanks to @haolibai

            #Original code:
            #grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
            #Change:
            #grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_weight * g)
            grad_alpha_temp =(indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())*a)
            grad_alpha=( grad_alpha_temp* grad_weight * g)
            grad_weight = indicate_middle * grad_weight
            print(" checkgr: ", grad_alpha[0][0])
            grad_thd=t.autograd.grad(grad_alpha[0][0], a)[0]*grad_alpha.mean()
            #print("grad is value: ", indicate_big)

            if indicate_big[0][0]==0 and indicate_small[0][0]==0:
                print("grad is value: ",grad_thd)
            else:
                print("grad is zero: ",grad_thd)

            return grad_weight, grad_alpha, grad_thd, None, None, None, None


class LSQ_prev(t.autograd.Function):
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
            grad_alpha_temp =(indicate_small * Qn + indicate_big * learn_thd + indicate_middle * (-q_w + q_w.round()))
            grad_alpha=( grad_alpha_temp* grad_weight * g)
            grad_weight = indicate_middle * grad_weight

            grad_thd=None
            #grad_thd=t.autograd.grad(grad_alpha.mean(), learn_thd)[0]*grad_alpha.mean()

            return grad_weight, grad_alpha, grad_thd, None, None, None, None

