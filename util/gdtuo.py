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

    def zero_grad(self):
        """ Set all gradients to zero. """

        self.module.zero_grad()

        for param in self.all_params_with_gradients:
            
            param.grad = torch.zeros_like(param)
        
        for param in self.a_params_with_gradients:
            param.grad = torch.zeros_like(param)

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


