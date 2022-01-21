import torch
import torch.nn as nn

from . import vgg


def skyline_model_provider():
    return vgg.vgg16().cuda()


def skyline_input_provider(batch_size=16):
    return (
        torch.randn((batch_size, 3, 224, 224)).cuda(),
        torch.randint(low=0, high=1000, size=(batch_size,)).cuda(),
    )


def skyline_iteration_provider(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    def iteration(*inputs):
        optimizer.zero_grad()
        out = model(*inputs)
        out = loss_fn(out, labels)
        out.backward()
        optimizer.step()
    return iteration
