import torch as t


class Quantizer(t.nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'
    def update_strname(self,strname):
        pass
    def update_list_for_lsq(self,num_solution,T):
        pass
    def forward(self, x):
        return x
