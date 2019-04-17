# pylint: disable=invalid-name,no-self-use,protected-access
from collections import namedtuple
import os
import pytest

from flaky import flaky
from numpy.testing import assert_almost_equal
import torch

from allennlp.common.testing import ModelTestCase

class EndToEndModuleNetworkTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "visual_question_answering" /
                              "end_to_end_module_network" / "experiment.json"),
                          str(self.FIXTURES_ROOT / "data" / "visual_question_answering" / "shapes"
                              / "train.tiny*"))

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
