# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.modules.image_encoders.conv_2d import Conv2d


class TestConv2d(AllenNlpTestCase):
    def test_can_construct_from_params(self):
        params = Params({'in_channels': 3,
                         'out_channels': 10,
                         'kernel_size': [3, 4],
                         'stride': 1,
                         'padding': 0,
                         'bias': False})
        conv_layer = Conv2d.from_params(params)
        assert conv_layer.in_channels == 3
        assert conv_layer.out_channels == 10
        assert conv_layer.kernel_size == (3, 4)
        assert conv_layer.stride == (1, 1)  # Conv2d makes this a tuple if you pass an int
        assert conv_layer.padding == (0, 0)  # this one too
        assert conv_layer.bias is None
