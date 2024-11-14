import torch
from quan.quantizer import lsq

class Optimizable:
    '''
    This is the interface for anything that has parameters that need to be
    optimized, somewhat like torch.nn.Model but with the right plumbing for
    hyperoptimizability. (Specifically, torch.nn.Model uses the Parameter
    interface which does not give us enough control about the detachments.)
    Nominal operation of an Optimizable at the lowest level is as follows:
        o = MyOptimizable(...)
        o.initialize()
        loop {
            o.begin()
            o.zero_grad()
            loss = --compute loss function from parameters--
            loss.backward()
            o.step()
        }
    Optimizables recursively handle updates to their optimiz*ers*.
    '''

    def __init__(self, parameters, optimizer, a_parameters , w_parameters,modules_to_quantize,excepts):
        self.parameters = parameters  # a dict mapping names to tensors
        self.optimizer = optimizer  # which must itself be Optimizable!
        self.all_params_with_gradients = []
        self.weight_params_with_gradients = []
        self.a_params_with_gradients = []
        self.meta_params_with_gradients = []
        self.a_params_names = []
        self.w_parameters = w_parameters
        self.a_parameters = a_parameters
        self.modules_to_quantize = modules_to_quantize
        self.excepts=excepts
    def initialize(self):
        ''' Initialize parameters, e.g. with a Kaiming initializer. '''
        pass

    def begin(self):
        ''' Enable gradient tracking on current parameters. '''
        for param in self.all_params_with_gradients:
            param.grad = None
        for param in self.a_params_with_gradients:     
            param.grad = None
        
        for param in self.meta_params_with_gradients:     
            param.grad = None

        self.all_params_with_gradients.clear()
        self.a_params_with_gradients.clear()
        for name, param in self.parameters.items():
            param.requires_grad_()  # keep gradient information...
            param.retain_grad()  # even if not a leaf...
            
            if name.endswith(".a"):
                self.a_params_with_gradients.append(param)
                self.a_params_names.append(name)
            else:
                if ("meta_modules" in name):
                    self.meta_params_with_gradients.append(param)
                else:
                    self.all_params_with_gradients.append(param)      
            
        self.optimizer.begin()

    def begin_w(self):
        ''' Enable gradient tracking on current parameters. '''
        for param in self.all_params_with_gradients:
            param.grad = None
        
        self.all_params_with_gradients.clear()
        temp_list=[]
        for name, param in self.parameters.items():
            param.requires_grad_()  # keep gradient information...
            param.retain_grad()  # even if not a leaf...
            
            if not name.endswith("a"):
                self.all_params_with_gradients.append(param)


    def zero_grad(self):
        ''' Set all gradients to zero. '''
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        
        for param in self.a_params_with_gradients:
            param.grad = torch.zeros_like(param)


        self.optimizer.zero_grad()

    def step(self):
        ''' Update parameters '''
        pass


class NoOpOptimizer(Optimizable):
    '''
    NoOpOptimizer sits on top of a stack, and does not affect what lies below.
    '''

    def __init__(self):
        pass

    def initialize(self):
        pass

    def begin(self):
        pass

    def zero_grad(self):
        pass

    def step(self, params):
        pass

    def __str__(self):
        return ''


class SGD(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''

    def __init__(self, alpha=0.01, mu=0.0, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu)
        }
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            if self.mu != 0.0:
                if name not in self.state:
                    buf = self.state[name] = g
                else:
                    buf = self.state[name].detach()
                    buf = buf * self.parameters['mu'] + g
                g = self.state[name] = buf
            params[name] = p - g * self.parameters['alpha']

    def __str__(self):
        return 'sgd / ' + str(self.optimizer)

def convert_name(name_gdtuo):
    str_name=name_gdtuo[name_gdtuo.find(".")+1:]
    str_name=str_name[:str_name.rfind(".")]
    if str_name.endswith('fn'):
        str_name=str_name[:str_name.rfind(".")]
    return str_name


class SGD_Delayed_Updates(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''

    def __init__(self, alpha=0.01, mu=0.0,alpha_for_a=0, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu),
            'alpha_for_a':torch.tensor(alpha_for_a)
        }
        self.eta={}
        self.save_f={}
        self.save_dfdLdV={}
        self.save_v_hat={}
        self.save_g={}
        self.final_wight_grad={}

        super().__init__(parameters, optimizer,None,parameters,{},{})

    def step_a(self, params,modules_to_quantize,excepts):# step on the a parameters
        self.optimizer.step(self.parameters)
        
        
        for name, param in params.items():
            if (convert_name(name) in modules_to_quantize.keys()) and name.endswith("quan_w_fn.a") and not(convert_name(name) in excepts.keys()):

                g=param.grad.detach()
                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf.detach() * self.parameters['mu'] + g
                    g = self.state[name] = buf
                
                params[name] = p - g * self.parameters['alpha_for_a']

                
    def step_w(self, params,modules_to_quantize,excepts):# step on the rest of the parametrs
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            if (convert_name(name) in modules_to_quantize.keys()) and name.endswith("weight")and not(convert_name(name) in excepts.keys()):# if this is a weight of a quantized    
                g = param.grad
                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf * self.parameters['mu'] + g
                    g = self.state[name] = buf
                params[name] = p - g * self.parameters['alpha']
            else:
                if not((convert_name(name) in modules_to_quantize.keys()) and name.endswith("quan_w_fn.a") and not(convert_name(name) in excepts.keys())):
                    
                    if not name.endswith("v_hat"):
                        
                        if param.grad != None:                                           
                            g=param.grad.detach()
                            p = param.detach()
                            if self.mu != 0.0:
                                if name not in self.state:
                                    buf = self.state[name] = g
                                else:
                                    buf = self.state[name].detach()
                                    buf = buf.detach() * self.parameters['mu'] + g
                                g = self.state[name] = buf
                            params[name] = p - g * self.parameters['alpha'].detach()


class SGD_less_greedy_Updates(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''

    def __init__(self, alpha=0.01, mu=0.0,alpha_for_a=0, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu),
            'alpha_for_a':torch.tensor(alpha_for_a)
        }
        self.eta={}
        self.save_f={}
        self.save_dfdLdV={}
        self.save_v_hat={}
        self.final_wight_grad={}
        self.save_g={}
        super().__init__(parameters, optimizer,None,parameters,{},{})

    def step_a(self, params,modules_to_quantize,excepts):# step on the a parameters
        self.optimizer.step(self.parameters)
        #print("stepping aaaaa")
        
        for name, param in params.items():
            if (convert_name(name) in modules_to_quantize.keys()) and name.endswith("quan_w_fn.a") and not(convert_name(name) in excepts.keys()):

                g=param.grad.detach()
                #print("g is : ",g)
                parts = name.split('.')
                result = '.'.join(parts[:-2])
                #print("results is ",result)
                #weight_param=params[result+'.weight']

                weight_grad=self.final_wight_grad[result+'.weight']

                
                self.save_g[result+'.weight'][-1].zero_()
                #print("weight.grad is : ",weight_grad)
                grads_div_a=torch.stack(self.save_g[result+'.weight'], dim=0)
                print("all dimenstions : "," weight_grad.detach() : ",weight_grad.unsqueeze(0).detach().shape," grads_div_a : ",grads_div_a.shape)
                total_grad= -self.parameters['alpha']*grads_div_a.detach()*weight_grad.unsqueeze(0).detach()
                #total_grad= self.parameters['alpha']*grads_div_a.detach()*weight_grad.unsqueeze(0).detach()
                
                #print()
                total_grad=torch.clamp(total_grad,1,-1)
                #print("total_grad : ",total_grad)
                if  torch.isnan(total_grad).any():
                    print("error7 here ",name,total_grad.detach()," weight_grad ",grads_div_a.detach())
                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf.detach() * self.parameters['mu'] + g
                    g = self.state[name] = buf
                #print("all dimenstions : "," p : ",p.detach().shape)
                #print("p is :",p)
                params[name] = p - total_grad.detach() * self.parameters['alpha_for_a']
                #print("bef last param a is : ",param[-1])
                #print("aft last param a is : ",params[name][-1])
                if  torch.isnan(params[name]).any():
                    print("error here ",name,total_grad.detach()," weight_grad ",weight_grad.detach()," g ",grads_div_a)
    
                
    def step_w(self, params,modules_to_quantize,excepts):# step on the rest of the parametrs
        hold_a_grad=None
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            #print(name)
            if (convert_name(name) in modules_to_quantize.keys()) and name.endswith("weight")and not(convert_name(name) in excepts.keys()):# if this is a weight of a quantized    
                #continue
                parts = name.split('.')
                result = '.'.join(parts[:-1])
                #print("results is ",result)
                x_hat_param=params[result+'.quan_w_fn.x_hat']
                if x_hat_param.grad!=None:
                    if True:#torch.any(x_hat_param.grad!=0):
                        #print("one")
                        second_order_derivative = torch.autograd.grad(x_hat_param.grad.sum(), param, retain_graph=True)
                        #print("second_order_derivative",second_order_derivative)
                        #torch.clamp(input_tensor, min_value, max_value)
                        #second_order_derivative = torch.autograd.grad(grad.sum(), w_param, retain_graph=True)
                        #print("second_order_derivative ",second_order_derivative)
                        ones_tensor=torch.ones(second_order_derivative[0].shape).cuda()
                        hold_a_grad=(ones_tensor-second_order_derivative[0]*self.parameters['alpha'])
                        if  torch.isnan(hold_a_grad).any():
                            print("error error important here ",name,hold_a_grad.detach())
                    else:
                        print("check if else")



                g = param.grad.detach()

                
                if name in self.save_g:
                    length=len(self.save_g[name])
                else:
                    length=0
                parts = name.split('.')
                result = '.'.join(parts[:-1])
                #print("results is ",result)
                a_params=params[result+'.quan_w_fn.a']
                a_param= a_params[length-1]
                #print("a_params.grad",a_params.grad)
                #print("sanity check(length) : ",torch.all(a_params.grad[length]==0)," and(length-1) : ",torch.all(a_params.grad[length-1]==0)," len : ",length)
                #print("sanity check(length) : ",torch.all(a_params.grad[-1]==0))
                
                if name in self.save_g:
                    #print("names here : ",name)
                    self.save_g[name].append(g.detach()/a_param.detach())
                else:
                    #print("names here2 : ",name)
                    self.save_g[name]=[]
                    self.save_g[name].append(g.detach()/a_param.detach())


                self.final_wight_grad[name]=param.grad.detach()
                if  torch.isnan(self.final_wight_grad[name]).any():
                    print("error97 erereroere ",name,self.final_wight_grad[name])
                p = param.detach()
                g = param.grad
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf * self.parameters['mu'] + g
                    g = self.state[name] = buf
                params[name] = p - g * self.parameters['alpha']

            else:
                if not((convert_name(name) in modules_to_quantize.keys()) and name.endswith("quan_w_fn.a") and not(convert_name(name) in excepts.keys())):
                    
                    if not name.endswith("v_hat"):
                        
                        if param.grad != None:                                           
                            g=param.grad.detach()
                            p = param.detach()
                            if self.mu != 0.0:
                                if name not in self.state:
                                    buf = self.state[name] = g
                                else:
                                    buf = self.state[name].detach()
                                    buf = buf.detach() * self.parameters['mu'] + g
                                g = self.state[name] = buf
                            params[name] = p - g * self.parameters['alpha'].detach()
                            
                else:
                    parts = name.split('.')
                    result = '.'.join(parts[:-2])
                    name_w=result+'.weight'
                    #print("a grad : ",p.grad)
                    if(hold_a_grad!= None):
                        if name_w in self.save_g:
                            #print("two")

                            #print("shape g : ",g.get_device(),"shape hold_a_grad : ",hold_a_grad.get_device())
                            #param.grad = ((g * hold_a_grad)/param.detach()).to(param.grad.dtype).to(param.grad.device)
                            #print(" nema ",name,name_w)
                            #print("shpae 1 : ",self.save_g[name_w][0].shape,"shpae 2 : ",hold_a_grad.shape )
                            
                            # print("checkckkckckc sfedsfdf ",self.save_g[name_w][-1])
                            # temp_save_g = [((tensor.detach() * hold_a_grad.detach())).to(param.grad.dtype).to(param.grad.device) for tensor in self.save_g[name_w][:-1]]
                            # print("self.save_g[name_w] bef ",self.save_g[name_w])
                            # self.save_g[name_w]=temp_save_g.append(self.save_g[name_w][-1])
                            # print("checke s df vds ",self.save_g[name_w][-1])
                            # print("self.save_g[name_w] aft ",self.save_g[name_w])
                            #print("check self.save_g[name_w][-1] bef",self.save_g[name_w][-1])
                            #print("check sizes self.save_g[name_w][0] : ",self.save_g[name_w][0].shape," sec hold_a_grad : ",hold_a_grad.shape)
                            self.save_g[name_w] = [
                                ((tensor.detach() * hold_a_grad.detach()).to(param.grad.dtype).to(param.grad.device)) 
                                if idx != len(self.save_g[name_w]) - 1 else tensor
                                for idx, tensor in enumerate(self.save_g[name_w])
                            ]
                            #print("check self.save_g[name_w][-1] aft",self.save_g[name_w][-1])
                            #self.save_g[name_w] = ((self.save_g[name_w] * hold_a_grad)).to(param.grad.dtype).to(param.grad.device)
                            if  any(torch.isnan(tensor).any() for tensor in self.save_g[name_w]):
                                print("error2 error2 important here ",name,param.grad)
                            #print(param.grad[500])
                        else:
                            print("ererererroror hererrrr5 : ",name)
                            self.save_g[name_w]=1
                        


class ModuleWrapper(Optimizable):
    '''
    This class tries to convert a torch.nn.Module to an Optimizable, handling
    the internal plumbing needed to update parameters correctly.
    '''

    def __init__(self, module, optimizer=NoOpOptimizer(),modules_to_quantize={},excepts={}):
        self.module = module
        parameters = {k: v for k, v in module.named_parameters(recurse=True)}
        a_params = []
        w_params = []
        for name, param in module.named_parameters(recurse=True):
            
            if name.endswith("a"):
                a_params.append(param)
            else:
                w_params.append(param)

        super().__init__(parameters, optimizer,a_params,w_params,modules_to_quantize,excepts)

    def initialize(self):
        self.optimizer.initialize()
    # def zero_grad_less_greedy(self):
    #     """ Set all gradients to zero. """

    #     self.module.zero_grad()

    #     for param in self.all_params_with_gradients:
            
    #         param.grad = torch.zeros_like(param)
        
    #     for param in self.a_params_with_gradients:
    #         param.grad = torch.zeros_like(param)
        
    #     self.optimizer.final_wight_grad.clear()
    #     self.optimizer.save_g.clear()
    #     self.optimizer.zero_grad()

    def zero_grad(self):
        """ Set all gradients to zero. """

        self.module.zero_grad()

        for param in self.all_params_with_gradients:
            
            param.grad = torch.zeros_like(param)
        
        for param in self.a_params_with_gradients:
            param.grad = torch.zeros_like(param)
        
        self.optimizer.final_wight_grad.clear()
        self.optimizer.save_g.clear()
        self.optimizer.zero_grad()

    def zero_grad_not_a(self):
        
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)

        for param in self.a_params_with_gradients:
            if param.grad != None:
                param.grad = param.grad.detach()
        
        
        self.optimizer.zero_grad()
             
        self.optimizer.zero_grad()


    def forward(self, *xyz):
        return self.module(*xyz)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()   

    def named_parameters(self):
        return self.module.named_parameters(recurse=True)

    def named_modules(self):
        return self.module.named_modules()

    def detach_params(self):
        for name, param in self.module.named_parameters():
            param=param.detach()
    def step_w(self):
        self.optimizer.step_w(self.parameters,self.modules_to_quantize,self.excepts)

        def set_param(m, k, v):
            kk = k
            while '.' in k:
                sm = k[:k.index('.')]
                k = k[k.index('.') + 1:]
                m = m._modules[sm]

            m._parameters[k] = None
            m._parameters[k] = self.parameters[kk]

        for k, v in self.module.named_parameters(recurse=True):
            set_param(self.module, k, v)

    def step_a(self):
        self.optimizer.step_a(self.parameters,self.modules_to_quantize,self.excepts)
        def set_param(m, k, v):
            kk = k
            while '.' in k:
                sm = k[:k.index('.')]
                k = k[k.index('.') + 1:]
                m = m._modules[sm]

            m._parameters[k] = None
            m._parameters[k] = self.parameters[kk]

        for k, v in self.module.named_parameters(recurse=True):
            set_param(self.module, k, v)


