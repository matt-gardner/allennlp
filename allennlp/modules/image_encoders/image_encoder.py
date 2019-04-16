from typing import Tuple

import torch
from overrides import overrides

from allennlp.common import Registrable


class ImageEncoder(torch.nn.Module, Registrable):
    """
    An ``ImageEncoder`` is a ``Module`` that takes as input an image and returns an encoded set of
    features.  Input shape: ``(batch_size, in_channels, height, width)`` (matching pytorch's Conv2d
    input shape); output shape: ``(batch_size, out_channels, encoded_height, encoded_width)``,
    where ``encoded_height`` and ``encoded_width`` do not necessarily match ``height`` and
    ``width``, due to pooling.

    We add two methods to the basic ``Module`` API: :func:`get_input_shape()` and
    :func:`get_output_shape()`.  You might need this if you want to, e.g., construct a ``Linear``
    layer using the output of this encoder, or to raise sensible errors for mis-matching input
    dimensions.
    """
    def get_input_channels(self) -> int:
        """
        Returns the number of channels in the input image that this ``ImageEncoder`` expects.
        """
        raise NotImplementedError

    def get_output_shape_for_input(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Returns the shape of the encoded features given an input shape.  Because the encoder likely
        doesn't require a fixed input image width and height, but does a fixed shrinking of those
        values, we require knowing the input shape in order to compute the output shape.  The input
        is expected to be ``(in_channels, in_height, in_width)``, and the output is
        ``(out_channels, out_height, out_width)``.
        """
        raise NotImplementedError


@ImageEncoder.register("pass_through")
class PassThroughImageEncoder(ImageEncoder):
    """
    Does no encoding, just passing through the input image representation as the "features".  This
    is just intended for testing purposes.
    """
    @overrides
    def get_input_channels(self) -> int:
        return 3

    @overrides
    def get_output_shape_for_input(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return input_shape

    @overrides
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image
