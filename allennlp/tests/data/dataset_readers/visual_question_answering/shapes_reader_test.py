# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import ShapesReader
from allennlp.common.testing import AllenNlpTestCase


class TestShapesReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = ShapesReader(data_split='train.tiny')
        data_directory = str(self.FIXTURES_ROOT / "data" / "visual_reasoning" / "shapes")
        instances = ensure_list(reader.read(data_directory))
        assert len(instances) == 64

        # All of the questions are the same
        expected_tokens = ['is', 'a', 'green', 'shape', 'left', 'of', 'a', 'red', 'shape']
        for instance in instances:
            assert [t.text for t in instance.fields["question"]] == expected_tokens

        # And all of the logical forms are the same: (exist (and_ (relocate find) find))
        expected_actions = ['@start@ -> Answer',
                            'Answer -> [<Attention:Answer>, Attention]',
                            '<Attention:Answer> -> exist',
                            'Attention -> [<Attention,Attention:Attention>, Attention, Attention]',
                            '<Attention,Attention:Attention> -> and_',
                            'Attention -> [<Attention:Attention>, Attention]',
                            '<Attention:Attention> -> relocate',
                            'Attention -> find',
                            'Attention -> find']
        for instance in instances:
            instance_action_map = {rule.rule: i for i, rule in enumerate(instance.fields['actions'].field_list)}
            expected_ids = [instance_action_map[action] for action in expected_actions]
            actual_ids = [f.sequence_index for f in instance.fields["target_action_sequence"].field_list]
            assert actual_ids == expected_ids

        # The images are all different, but we'll just test the shape, anyway.
        for instance in instances:
            assert instance.fields['image'].array.shape == (30, 30, 3)

        # And lastly, the answers are all different; we'll just test the first few.
        assert instances[0].fields['answer'].label == 'false'
        assert instances[1].fields['answer'].label == 'true'
        assert instances[2].fields['answer'].label == 'false'
        assert instances[3].fields['answer'].label == 'false'
        assert instances[4].fields['answer'].label == 'false'
        assert instances[5].fields['answer'].label == 'true'
