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
        #print("check a prarm list : ",self.a_params_with_gradients)
        for param in self.a_params_with_gradients:
            
            
                        
            #print("flag here tttt")
            param.grad = None
        self.all_params_with_gradients.clear()
        self.a_params_with_gradients.clear()
        for name, param in self.parameters.items():
            #print("name in begin : ",name)
            param.requires_grad_()  # keep gradient information...
            param.retain_grad()  # even if not a leaf...
            
            if name.endswith(".a"):
                #print("at begin in mw a mean is : ",param.mean()," a var is : ",param.var())
                self.a_params_with_gradients.append(param)
                self.a_params_names.append(name)
            else:
                self.all_params_with_gradients.append(param)
            
            #print("name here : ",name)
        #print("size is here : ",len(self.all_params_with_gradients))
        self.optimizer.begin()

    def begin_w(self):
        ''' Enable gradient tracking on current parameters. '''
        #print("check size of a bef : ",len(self.a_params_with_gradients))
        #print("cehck a params with grad 1 : ",self.a_params_with_gradients[0].grad)
        # for i in range(len(self.a_params_with_gradients)):
        #     if self.a_params_with_gradients[i].grad == None:
        #         #print("check grad : ",self.a_params_with_gradients[i].grad)
        #         #print("check name : ",self.a_params_names[i])
        #         pass
        #     else:
        #         if self.a_params_with_gradients[i].grad[100].mean() !=0:
        #             print("check grad : ",self.a_params_with_gradients[i].grad[100].mean())
        #             #print("check size : ",len(self.a_params_with_gradients[i].grad))
        #             print("check name : ",self.a_params_names[i])
        #print("check a prarm list : ",self.a_params_with_gradients)
        for param in self.all_params_with_gradients:
            param.grad = None
        
        self.all_params_with_gradients.clear()
        #print("cehck a params with grad 2 : ",self.a_params_with_gradients[0].grad)
        #print("check size of a aft : ",all(v is None for v in self.a_params_with_gradients))
        temp_list=[]
        for name, param in self.parameters.items():
            #print(param.grad)
            #print("name in begin : ",name)
            param.requires_grad_()  # keep gradient information...
            param.retain_grad()  # even if not a leaf...
            
            if not name.endswith("a"):
                self.all_params_with_gradients.append(param)
                #if(param != None):
                #    print("name of param ",name," param to check if accumaltes",param.mean())
        

        #print(self.a_params_with_gradients)
        #self.all_params_with_gradients.extend(self.a_params_with_gradients)
                
            #print("name here : ",name)
        #print("size is here : ",len(self.all_params_with_gradients))


    def zero_grad(self):
        ''' Set all gradients to zero. '''
        #print("size is here real : ",len(self.all_params_with_gradients))
        for param in self.all_params_with_gradients:
            
            param.grad = torch.zeros_like(param)
            #print("param ",param.grad)
        
        for param in self.a_params_with_gradients:
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

def convert_name(name_gdtuo):
    str_name=name_gdtuo[name_gdtuo.find(".")+1:]
    str_name=str_name[:str_name.rfind(".")]
    if str_name.endswith('fn'):
        str_name=str_name[:str_name.rfind(".")]
    return str_name

class SGD_for_gSTE(Optimizable):
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
            #print(" check name : ",name)
            #print(" check convert name : ",convert_name(name))
            #print(" check convert modules_to_quantize.keys() : ",modules_to_quantize.keys())
            #if (convert_name(name) in excepts.keys()):
            #    print(" full excpets : ", excepts)
            #    print("check excepts : ",name)
            if (convert_name(name) in modules_to_quantize.keys()) and name.endswith("quan_w_fn.a") and not(convert_name(name) in excepts.keys()):
                #print("names that got in : ",name)
                if not (name in self.eta.keys()):
                    self.eta.update({name:torch.ones(param.size())*self.parameters["alpha_for_a"]})
                    #print(self.eta[name])
                p = param.detach()
                #print("check params: ",p)
                c = save_gradients.get_dfdLdV(convert_name(name)).cuda()*save_gradients.get_dLdV_hat(convert_name(name)).cuda()*lsq.save_gradients.get_grad(convert_name(name)).cuda()*lsq.save_gradients.get_mult(convert_name(name)).cuda()
                #print(" check c elements : ",c.mean())
                
                #if name.endswith("quan_a_fn.a"):
                #    print(" check c of quan_a_fn.a : ",c)
                vdivs_Qq = lsq.save_gradients.get_vdivs(convert_name(name)).cuda()        

                both_hold = torch.where( torch.logical_and((p.ge(vdivs_Qq) ) , (((1/(1-self.eta[name].cuda() * c ))*p).lt(vdivs_Qq))) , 1 , 0)
                none_hold = torch.where( torch.logical_and((torch.logical_not((p.ge(vdivs_Qq) ))) , (torch.logical_not(((1/(1-self.eta[name].cuda() * c ))*p).lt(vdivs_Qq)))) , 1 , 0)
                check=0
                while both_hold.sum() > 0 and none_hold.sum()>0:
                    #print(self.eta[name])
                    self.eta[name] = torch.where(torch.logical_or(both_hold, none_hold),self.eta[name].detach().cuda()/2,self.eta[name].detach().cuda())
                    
                    both_hold = torch.where( torch.logical_and((p.ge(vdivs_Qq)) , (((1/(1-self.eta[name].cuda() * c ))*p).lt(vdivs_Qq))) , 1 , 0)
                    none_hold = torch.where( torch.logical_and((torch.logical_not(p.ge(vdivs_Qq))) , (torch.logical_not(((1/(1-self.eta[name].cuda() * c ))*p).lt(vdivs_Qq)))) , 1 , 0)

                    #if check>=1:
                    #    print(self.eta[name])
                    #check+=1

                params[name] = torch.where(p.ge(vdivs_Qq) , p , ((1/(1-self.eta[name].cuda() * c ))*p).detach())
                #check = torch.where(p.ge(vdivs_Qq) , 1 , 2)
                #print("check params: ",params[name])
                
    
    def step_w(self, params,modules_to_quantize,excepts):# step on the rest of the parametrs
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            #print("names in step_w : ",name)
            if (convert_name(name) in modules_to_quantize.keys()) and name.endswith("weight")and not(convert_name(name) in excepts.keys()):# if this is a weight of a quantized 
                
                
                grad_tesnor = param.grad.detach().requires_grad_()
                g=grad_tesnor

                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf.detach() * self.parameters['mu'] + g
                    g = self.state[name] = buf
                f_optimizer = g * self.parameters['alpha']
                params[name] = p - g.detach() * self.parameters['alpha'].detach()


                f_optimizer.sum().backward()
                save_gradients.update_dfdLdV(convert_name(name),grad_tesnor.grad)
            else:
                if not((convert_name(name) in modules_to_quantize.keys()) and name.endswith("quan_w_fn.a") and not(convert_name(name) in excepts.keys())):
                    if not name.endswith("v_hat"):
                        #print("check name : ",name)
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

                    

    def __str__(self):
        return 'sgd / ' + str(self.optimizer)


class save_gradients():
    save_dfdLdV={}
    save_dLdV_hat={}
    @staticmethod
    def get_dfdLdV(name):
        return save_gradients.save_dfdLdV[name]

    @staticmethod
    def update_dfdLdV(name, dfdLdV):
        
        save_gradients.save_dfdLdV.update({name: dfdLdV.detach()})

    @staticmethod
    def get_dLdV_hat(name):
        return save_gradients.save_dLdV_hat[name]

    @staticmethod
    def update_dLdV_hat(name, dLdV_hat):
        #print(" check names here ",name)
        save_gradients.save_dLdV_hat.update({name: dLdV_hat.clone()})



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
        #for name, param in params.items():
        #    if param.grad.sum() !=0:
        #        print("check if grad zero ",name," val: ",param.grad)
        
        for name, param in params.items():
            if (convert_name(name) in modules_to_quantize.keys()) and name.endswith("quan_w_fn.a") and not(convert_name(name) in excepts.keys()):
                #print("param is : ",param.mean())

                g=param.grad.detach()
                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf.detach() * self.parameters['mu'] + g
                    g = self.state[name] = buf
                #print("check if grad zero ",params[name])
                
                params[name] = p - g * self.parameters['alpha_for_a']
                #print("new param is : ",params[name].mean())

                #print("len params : ",len(params))

                
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
                #if g ==None:
                #    print("check g none : ",name)
                #else:
                #print("grad of a : ",param.grad)
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
                        #else:
                        #    print(" check name param ",name,param.grad)
                #else:
                #    if param.grad[100].mean() != 0:
                #           print(" check grad name ",name,param.grad[100].mean())


class SGD_Delayed_Updates_meta(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''

    def __init__(self, alpha=0.01, mu=0.0,alpha_for_a=0,alpha_for_b=0, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu),
            'alpha_for_a':torch.tensor(alpha_for_a),
            'alpha_for_b':torch.tensor(alpha_for_b)
        }
        self.eta={}
        self.save_f={}
        self.save_dfdLdV={}
        self.save_v_hat={}
        super().__init__(parameters, optimizer,None,parameters,{},{})

    def step_a_and_b(self, params,modules_to_quantize,excepts):# step on the a parameters
        self.optimizer.step(self.parameters)
        #for name, param in params.items():
        #    if param.grad.sum() !=0:
        #        print("check if grad zero ",name," val: ",param.grad)
        
        for name, param in params.items():
            if (convert_name(name) in modules_to_quantize.keys()) and (name.endswith("quan_w_fn.a") or name.endswith("quan_w_fn.b")) and not(convert_name(name) in excepts.keys()):
                #print("check name : ",name)
                #print("param is : ",param.mean())

                g=param.grad.detach()
                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf.detach() * self.parameters['mu'] + g
                    g = self.state[name] = buf
                #print("check if grad zero ",params[name])
                if name.endswith("quan_w_fn.a"):
                    params[name] = p - g * self.parameters['alpha_for_a']
                else:
                    #print("param is : ",param.mean())
                    params[name] = p - g * self.parameters['alpha_for_b']
                #print("new param is : ",params[name].mean())

                #print("len params : ",len(params))

                
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
                #if g ==None:
                #    print("check g none : ",name)
                #else:
                #print("grad of a : ",param.grad)
                params[name] = p - g * self.parameters['alpha']
            else:
                if not(convert_name(name) in modules_to_quantize.keys()) and (name.endswith("quan_w_fn.a") or name.endswith("quan_w_fn.b")) and not(convert_name(name) in excepts.keys()):
                    
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
                        #else:
                        #    print(" check name param ",name,param.grad)
                #else:
                #    print(" check grad name ",name,param.grad[4].sum())
                    
                        


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

    def __init__(self, module, optimizer=NoOpOptimizer(),modules_to_quantize={},excepts={}):
        self.module = module
        parameters = {k: v for k, v in module.named_parameters(recurse=True)}
        a_params = []
        w_params = []
        #a_params = []
        #w_params = []
        for name, param in module.named_parameters(recurse=True):
            
            if name.endswith("a"):
                #print("inside mw a mean is : ",param.mean()," a var is : ",param.var())
                a_params.append(param)
                #a_params.append( param)

            else:
                w_params.append(param)

        super().__init__(parameters, optimizer,a_params,w_params,modules_to_quantize,excepts)

    def initialize(self):
        self.optimizer.initialize()
    def get_dl_dv_hat(self):
        for name in self.parameters:
            if name.endswith("v_hat"):
                if not "quan_a_fn" in name:
                    save_gradients.update_dLdV_hat(convert_name(name),self.parameters[name].grad.detach())

    def check_grad_vals(self):
        for name in self.parameters:
            if self.parameters[name].grad == None:
                print(" name is : ",name," chekc grad sum : ",self.parameters[name].grad)
            else:
                print(" name is : ",name," chekc grad sum : ",self.parameters[name].grad.sum())

            
        


    def zero_grad(self):
        """ Set all gradients to zero. """
        #print("size is here real : ",len(self.all_params_with_gradients))

        self.module.zero_grad()
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        
        for param in self.a_params_with_gradients:
            param.grad = torch.zeros_like(param)

        self.optimizer.zero_grad()

    def zero_grad_not_a(self):
        # for i in range(len(self.a_params_with_gradients)):
        #     if self.a_params_with_gradients[i].grad == None:
        #         #print("check grad : ",self.a_params_with_gradients[i].grad)
        #         #print("check name : ",self.a_params_names[i])
        #         pass
        #     else:
        #         if self.a_params_with_gradients[i].grad[100].mean() !=0:
        #             print("check grad zero grad : ",self.a_params_with_gradients[i].grad[100].mean())
        #             #print("check size : ",len(self.a_params_with_gradients[i].grad))
        #             print("check name zero grad : ",self.a_params_names[i])
        
        
        #self.module.zero_grad()

        
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)

        for param in self.a_params_with_gradients:
            if param.grad != None:
                param.grad = param.grad.detach()
        
        
        self.optimizer.zero_grad()
        
    def forward(self, *xyz):
        return self.module(*xyz)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()   

    def named_modules(self):
        return self.module.named_modules()

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


    def step_a_and_b(self):
        self.optimizer.step_a_and_b(self.parameters,self.modules_to_quantize,self.excepts)

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

