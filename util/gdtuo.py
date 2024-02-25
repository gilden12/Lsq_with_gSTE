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

    def __init__(self, parameters, optimizer, a_parameters , w_parameters,modules_to_quantize):
        self.parameters = parameters  # a dict mapping names to tensors
        self.optimizer = optimizer  # which must itself be Optimizable!
        self.all_params_with_gradients = []
        self.w_parameters = w_parameters
        self.a_parameters = a_parameters
        self.modules_to_quantize = modules_to_quantize
    def initialize(self):
        ''' Initialize parameters, e.g. with a Kaiming initializer. '''
        pass

    def begin(self):
        ''' Enable gradient tracking on current parameters. '''
        for param in self.all_params_with_gradients:
            param.grad = None
        self.all_params_with_gradients.clear()
        for name, param in self.parameters.items():
            param.requires_grad_()  # keep gradient information...
            param.retain_grad()  # even if not a leaf...
            self.all_params_with_gradients.append(param)
        self.optimizer.begin()

    def zero_grad(self):
        ''' Set all gradients to zero. '''
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()

    ''' Note: at this point you would probably call .backwards() on the loss
    function. '''

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

class SGD_for_gSTE(Optimizable):
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
        super().__init__(parameters, optimizer,None,parameters,{})

    def step(self, params,modules_to_quantize):
        print("here 3")
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            if name.endswith("a"):
                
                #print("check what did a ")
                g = param.grad.detach()
                print(" g for a is : ",g)
                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf * self.parameters['mu'] + g
                    g = self.state[name] = buf
                params[name] = p - g * self.parameters['alpha']
        
        for name, param in params.items():
            #print("check if name in ",name in modules_to_quantize)
            #print(name)
            #print("now mod to quan",modules_to_quantize.keys())
            if name in modules_to_quantize:
                #print("check what did w ")
                g = param.grad
                #print("name of tensor : ",name,"value fo g : ",g)
                p = param.detach()
                # if self.mu != 0.0:
                #     if name not in self.state:
                #         buf = self.state[name] = g
                #     else:
                #         buf = self.state[name].detach()
                #         buf = buf.detach() * self.parameters['mu'] + g
                #     g = self.state[name] = buf
                params[name] = p - g * self.parameters['alpha']
        
        

    def __str__(self):
        return 'sgd / ' + str(self.optimizer)
    
class SGD_for_a(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''

    def __init__(self, alpha=0.01, mu=0.0, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu),
            'a':torch.tensor()
        }
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            #dV_hatdV = Gradient_Calculation(a,Qp,Qn,SdivV)
            grad=param.grad.detach()
            g = grad
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



class SGDPerParam(Optimizable):
    '''
    Optimizes parameters individually with SGD.
    '''

    def __init__(self, params, optimizer=NoOpOptimizer()):
        parameters = {k + '_alpha': torch.tensor(v) for k, v in params}
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            if name + '_alpha' not in self.parameters:
                params[name] = p
            else:
                params[name] = p - g * self.parameters[name + '_alpha']

    def __str__(self):
        return 'sgdPerParam / ' + str(self.optimizer)



class ModuleWrapper(Optimizable):
    '''
    This class tries to convert a torch.nn.Module to an Optimizable, handling
    the internal plumbing needed to update parameters correctly.
    '''

    def __init__(self, module, optimizer=NoOpOptimizer(),modules_to_quantize={}):
        self.module = module
        parameters = {k: v for k, v in module.named_parameters(recurse=True)}
        a_params = {}
        w_params = {}
        #a_params = []
        #w_params = []
        for name, param in module.named_parameters(recurse=True):
            
            if name.endswith("a"):
                a_params.update({name: param})
                #a_params.append( param)

            else:
                w_params.update({name: param})
        new_dict={}
        count=0
        #print("cehck before : ",modules_to_quantize)
        for k, v in modules_to_quantize.items():
            if count>0:
            #print("check if it worggsgsgsgsgsg : "+"module."+k+".weight")
                new_dict["module."+k+".weight"] = k
            else:
                count+=1
        #print("cehck end : ",new_dict.keys())
        #print("dafaq ",w_params.keys())


        super().__init__(parameters, optimizer,a_params,w_params,new_dict)

    def initialize(self):
        self.optimizer.initialize()

    def zero_grad(self):
        """ Set all gradients to zero. """
        self.module.zero_grad()
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()

    def forward(self, *xyz):
        return self.module(*xyz)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def step(self):
        self.optimizer.step(self.parameters,self.modules_to_quantize)

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

        # for k, v in self.a_parameters.items():
        #     print("meanwhile 1 ", len(self.a_parameters))
        #     set_param(self.module, k, v)

        # for k, v in self.w_parameters.items():
        #     print("meanwhile 2 ", len(self.w_parameters))

        #     set_param(self.module, k, v)