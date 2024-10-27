import torch.nn as nn
import torch
import numpy as np
from abc import abstractmethod
from typing import Any, TypeVar, Iterator, Iterable, Generic
import copy

class Module(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters]) # np.prod 连乘
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def _forward_unimplemented(self, *input: Any) -> None:
        # To stop PyTorch from giving abstract methods warning
        pass

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get('__call__', None) is None:
            return

        setattr(cls, 'forward', cls.__dict__['__call__'])
        delattr(cls, '__call__')

    @property
    def device(self):
        params = self.parameters()
        try:
            sample_param = next(params)
            return sample_param.device
        except StopIteration:
            raise RuntimeError(f"Unable to determine"
                               f" device of {self.__class__.__name__}") from None


if __name__ == '__main__':
    m = Module()
    print(m.device)