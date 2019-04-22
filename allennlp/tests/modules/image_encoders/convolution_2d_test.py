# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.modules.image_encoders.convolution_2d import Convolution2d


class TestConvolution2d(AllenNlpTestCase):
    def test_can_construct_from_params(self):
        params = Params({'in_channels': 3,
                         'out_channels': 10,
                         'kernel_size': [3, 4],
                         'stride': 1,
                         'padding': 0,
                         'bias': False})
        conv_layer = Convolution2d.from_params(params)
        assert conv_layer.in_channels == 3
        assert conv_layer.out_channels == 10
        assert conv_layer.kernel_size == (3, 4)
