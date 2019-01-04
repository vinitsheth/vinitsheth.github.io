---
layout: post
title: "Python Syntax: “Callable Object”"
excerpt: "Python Learning Notes from a PyTorch Syntax"
modified: 2018-1-20T14:17:25-04:00
categories: blog
tags: [Python, PyTorch]
comments: true
share: true
---

### Introduction

Due to the intructor of the TTIC 31230 Deep Learning course requires us to use PyTorch for this course, instead of keep using TensorFlow, I have to start to learn PyTorch recently.

### Problem

When I was reading an elementary [PyTorch code](http://pytorch.org/docs/master/nn.html), I feel confused.

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
```

Since ```self.conv1``` is a object, how could it take an input directly when it is not for initialization (```self.conv1(x)```)?

### Solution

Then the first thing I did is to check the source code of ```nn.Conv2d```.

```python
class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

```

Clearly ```self.conv1(x)``` is doing ```self.conv1.forward(x)```. But how could we do ```self.conv1(x)```?

<br />

I trace back to the source code of the inherited objects [```_ConvNd```](http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d) and [```Module```](http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module.forward).


```python
class _ConvNd(Module):
...
```


```python
class Module(object):
...
    def __call__(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        if torch.jit._tracing:
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs) # Run self.forward !
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, Variable):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, Variable)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result
...
```

It turns out that Python class object has a ```__call__``` feature which I have never used before. The official documentation of this could be found [here](https://docs.python.org/3/reference/datamodel.html#object.__call__). Basically the content of ```__call__``` is run when the class object is "called" directly. We clearly see that there is ```result = self.forward(*input, **kwargs)``` in the ```__call__``` feature, which explains why we can do ```self.conv1(x)``` directly.

### Conclusion

There is still a lot to learn Python. Keep reading and writing codes!

