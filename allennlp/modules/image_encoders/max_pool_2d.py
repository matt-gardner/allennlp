from typing import Tuple, Union

import torch

from allennlp.common.from_params import FromParams


class MaxPool2d(torch.nn.MaxPool2d, FromParams):
    """
    A simple wrapper around ``torch.nn.MaxPool2d`` that lets us construct this ``from_params``.
    """
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False) -> None:
        super().__init__(kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         return_indices=return_indices,
                         ceil_mode=ceil_mode)

    def get_output_shape_for_input(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Returns the shape of the convolved features given an input shape.  The input is expected to
        be ``(in_channels, in_height, in_width)``, and the output is ``(out_channels, out_height,
        out_width)``.
        """
        in_channels, in_height, in_width = input_shape
        padding = [self.padding] * 2 if isinstance(self.padding, int) else self.padding
        dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation
        kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        # Equations taken from here: https://pytorch.org/docs/stable/nn.html#maxpool2d
        out_height = int(
                (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
                )
        out_width = int(
                (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
                )
        return in_channels, out_height, out_width
