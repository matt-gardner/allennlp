from typing import List, Tuple, Union

import torch
from overrides import overrides

from allennlp.common import Registrable
from allennlp.modules.image_encoders.conv_2d import Conv2d
from allennlp.modules.image_encoders.image_encoder import ImageEncoder
from allennlp.modules.image_encoders.max_pool_2d import MaxPool2d
from allennlp.nn import Activation


@ImageEncoder.register("convolutional")
class ConvolutionalImageEncoder(ImageEncoder):
    """
    An ``ImageEncoder`` that's a stacked series of convolutional layers, with some activation in
    between.  This is not pretrained, or fancy about anything, and is only really intended for
    small, toy image tasks.  For a real computer vision task, you almost certainly want a
    pre-trained ResNet or something better than this.
    """
    def __init__(self,
                 conv_layers: List[Conv2d],
                 activations: Union[Activation, List[Activation]],
                 max_pool_layers: List[MaxPool2d] = None,
                 dropout: Union[float, List[float]] = 0.0) -> None:
        super().__init__()
        if not isinstance(activations, list):
            activations = [activations] * len(conv_layers)  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * len(conv_layers)  # type: ignore
        if not isinstance(max_pool_layers, list):
            max_pool_layers = [max_pool_layers] * len(conv_layers)
        self._conv_layers = torch.nn.ModuleList(conv_layers)
        self._pool_layers = torch.nn.ModuleList(max_pool_layers)
        self._activations = activations
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)

    def get_input_channels(self) -> int:
        return self._conv_layers[0].in_channels

    def get_output_shape_for_input(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Returns the shape of the encoded features given an input shape.  Because the encoder likely
        doesn't require a fixed input image width and height, but does a fixed shrinking of those
        values, we require knowing the input shape in order to compute the output shape.  The input
        is expected to be ``(in_channels, in_height, in_width)``, and the output is
        ``(out_channels, out_height, out_width)``.
        """
        result_shape = input_shape
        for conv_layer, pool_layer in zip(self._conv_layers, self._pool_layers):
            result_shape = conv_layer.get_output_shape_for_input(result_shape)
            if pool_layer:
                result_shape = pool_layer.get_output_shape_for_input(result_shape)
        return result_shape

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        # pylint: disable=arguments-differ
        result = image
        for conv_layer, pool_layer, activation in zip(self._conv_layers,
                                                      self._pool_layers,
                                                      self._activations):
            result = conv_layer(result)
            if pool_layer:
                result = pool_layer(result)
            result = activation(result)
        return result
