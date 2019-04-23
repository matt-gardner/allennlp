# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.modules.image_encoders import ConvolutionalImageEncoder


class TestConvolutionalImageEncoder(AllenNlpTestCase):
    def test_can_construct_from_params(self):
        params = Params({
                'conv_layers': [
                        {'in_channels': 3,
                         'out_channels': 10,
                         'kernel_size': 3,
                         'padding': 1,
                         },
                        {'in_channels': 10,
                         'out_channels': 2,
                         'kernel_size': 3,
                         'padding': 1,
                         },
                        ],
                'activations': 'relu',
                'max_pool_layers': [
                        {'kernel_size': 2, 'stride': 2},
                        None,
                        ],
                'dropout': .2,
                })
        encoder = ConvolutionalImageEncoder.from_params(params)
        input_shape = (4, 3, 6, 12)  # (batch, in_channels, height, width)
        expected_output_shape = (4, 2, 3, 6)  # (batch, out_channels, out_height, out_width)

        # This function operates without the batch dimension.
        assert encoder.get_output_shape_for_input(input_shape[1:]) == expected_output_shape[1:]

        image = torch.rand(*input_shape)
        image_features = encoder(image)
        assert image_features.size() == expected_output_shape
