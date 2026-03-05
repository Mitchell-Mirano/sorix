import pytest
import numpy as np
import sorix
from sorix import Tensor, tensor
from sorix.nn import Module, Sequential, Linear, ReLU

def test_module_forward_not_implemented():
    m = Module()
    with pytest.raises(NotImplementedError, match="You must implement forward"):
        m(tensor([1.0]))

def test_module_repr_and_indent():
    class SubModule(Module):
        def forward(self, x): return x
        def extra_repr(self): return "info=test"

    class MainModule(Module):
        def __init__(self):
            super().__init__()
            self.sub = SubModule()
            self.lin = Linear(2, 1)
        def forward(self, x): return x

    m = MainModule()
    r = repr(m)
    assert "SubModule(info=test)" in r
    
    # Test multi-line extra_repr
    class MultiLineModule(Module):
        def extra_repr(self): return "line1\nline2"
    assert "line1\n  line2" in repr(MultiLineModule())
    
    # Test Sequential repr
    seq = Sequential(Linear(1, 1))
    assert "Sequential(" in repr(seq)
    assert "(0): Linear(" in repr(seq)
    
    # Test base Module repr (no children, no extra_repr)
    assert repr(Module()) == "Module()"
    
    # Test __str__
    assert str(m) == r

def test_module_to_non_module_attr():
    class PlainObj:
        def __init__(self):
            self.t = tensor([1.0, 2.0])
            
    class CustomModule(Module):
        def __init__(self):
            super().__init__()
            self.plain = PlainObj()
            
    m = CustomModule()
    # Mock cuda move by checking the 'to' call if we can't actually use cuda
    # But sorix .to('cpu') still calls _apply
    m.to('cpu')
    assert m.plain.t.device == 'cpu'

def test_module_state_dict_edge_cases():
    class CustomModule(Module):
        def __init__(self):
            super().__init__()
            self.p = tensor([1.0], requires_grad=True)
            self.l1 = Linear(1, 1)

    m = CustomModule()
    state = m.state_dict()
    
    # Extra key in state_dict should be ignored
    state['extra'] = tensor([5.0])
    m.load_state_dict(state) # Should ignore 'extra'
    
    sub_state = {"p": tensor([3.0]), "l1.W": tensor([[3.0]])}
    m.load_state_dict(sub_state)
    assert m.p.item() == 3.0
    
    # Non-tensor in state_dict (just to hit the branch)
    m.load_state_dict({"p": 123})
    assert m.p.item() == 3.0 # No change

def test_module_state_dict_recursion_non_module():
    class Sub:
        def __init__(self):
            self.t = tensor([5.0])
    class Root(Module):
        def __init__(self):
            super().__init__()
            self.sub_obj = Sub()
    m = Root()
    sd = m.state_dict()
    assert "sub_obj.t" in sd

def test_module_train_eval_modes():
    class Sub(Module):
        def forward(self, x): return x
    class Root(Module):
        def __init__(self):
            super().__init__()
            self.s = Sub()
            self.layers = [Sub()]
            self.d = {"a": Sub()}
    
    m = Root()
    m.eval()
    assert not m.training
    assert not m.s.training
    assert not m.layers[0].training
    assert not m.d["a"].training
    
    m.train()
    assert m.training
    assert m.s.training
    assert m.layers[0].training
    assert m.d["a"].training

def test_sequential_forward():
    s = Sequential(Linear(2, 2), ReLU(), Linear(2, 1))
    x = tensor([[1.0, 2.0]])
    out = s(x)
    assert out.shape == (1, 1)

def test_module_to_recursion():
    class Sub(Module):
        def forward(self, x): return x
    class Root(Module):
        def __init__(self):
            super().__init__()
            self.l = [Sub()]
            self.t = (Sub(),)
            self.d = {"a": Sub()}
    m = Root()
    m.to('cpu')
    assert m.l[0].device == 'cpu'
    assert m.t[0].device == 'cpu'
    assert m.d["a"].device == 'cpu'

def test_sequential_advanced_indexing():
    s = Sequential(Linear(2, 2), ReLU(), Linear(2, 1))
    
    # String index
    assert s['0'] is s[0]
    assert s['1'] is s[1]
    
    # Negative index
    assert s[-1] is s[2]
    assert s[-2] is s[1]
    
    # Slice
    sub = s[0:2]
    assert isinstance(sub, Sequential)
    assert len(sub) == 2
    assert sub[0] is s[0]

def test_sequential_iteration():
    layers = [Linear(2, 2), ReLU()]
    s = Sequential(*layers)
    
    # List comprehension uses __iter__
    iter_layers = [l for l in s]
    assert len(iter_layers) == 2
    assert iter_layers[0] is layers[0]
    assert iter_layers[1] is layers[1]

def test_module_nested_parameter_collection():
    class DeepNet(Module):
        def __init__(self):
            super().__init__()
            self.layers = [Sequential(Linear(2, 2)), [Linear(2, 1)]]
            self.more = {"a": Linear(1, 1)}

    net = DeepNet()
    params = net.parameters()
    # Sequential(Linear: 2) + List(Linear: 2) + Dict(Linear: 2) = 6
    assert len(params) == 6

def test_add_indent_internal():
    from sorix.nn.net import _add_indent
    s = "line1\nline2"
    indented = _add_indent(s, 2)
    assert indented == "line1\n  line2"
    
    single = "single"
    assert _add_indent(single, 2) == "single"
