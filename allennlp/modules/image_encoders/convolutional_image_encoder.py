from typing import Tuple

import torch
from overrides import overrides

from allennlp.common import Registrable


class ConvolutionalImageEncoder(ImageEncoder):
    """
    An ``ImageEncoder`` that's a stacked series of convolutional layers, with some activation in
    between.  This is not pretrained, or fancy about anything, and is only really intended for
    small, toy image tasks.  For a real computer vision task, you almost certainly want a
    pre-trained ResNet or something better than this.
    """
    def __init__(self,
                 conv_layers: List[Convolution2d],
                 activations: Union[Activation, List[Activation]],
                 dropout: Union[float, List[float]] = 0.0) -> None:
        self.conv_layers = conv_layers

    def get_input_channels(self) -> int:
        return 3

    def get_output_shape_for_input(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Returns the shape of the encoded features given an input shape.  Because the encoder likely
        doesn't require a fixed input image width and height, but does a fixed shrinking of those
        values, we require knowing the input shape in order to compute the output shape.  The input
        is expected to be ``(in_channels, in_height, in_width)``, and the output is
        ``(out_channels, out_height, out_width)``.
        """
        raise NotImplementedError
