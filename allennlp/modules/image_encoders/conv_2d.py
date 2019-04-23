from typing import Tuple, Union

import torch

from allennlp.common.from_params import FromParams


class Conv2d(torch.nn.Conv2d, FromParams):
    """
    A simple wrapper around ``torch.nn.Conv2d`` that lets us construct this ``from_params``.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)

    def get_output_shape_for_input(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Returns the shape of the convolved features given an input shape.  The input is expected to
        be ``(in_channels, in_height, in_width)``, and the output is ``(out_channels, out_height,
        out_width)``.
        """
        in_channels, in_height, in_width = input_shape
        # Equations taken from here: https://pytorch.org/docs/stable/nn.html#conv2d
        out_height = int(
                (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                / self.stride[0]
                + 1
                )
        out_width = int(
                (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                / self.stride[1]
                + 1
                )
        return self.out_channels, out_height, out_width
