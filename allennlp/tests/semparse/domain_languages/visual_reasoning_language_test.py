import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.domain_languages import VisualReasoningLanguage
from allennlp.semparse.domain_languages.visual_reasoning_language import VisualReasoningParameters
from allennlp.tests.semparse.domain_languages.domain_language_test import check_productions_match


class VisualReasoningLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.image_height = 4
        self.image_width = 5
        self.image_encoding_dim = 3
        self.text_encoding_dim = 6
        self.hidden_dim = 6
        self.num_answers = 7
        self.image_features = torch.rand(self.image_height, self.image_width, self.image_encoding_dim)
        self.parameters = VisualReasoningParameters(image_height=self.image_height,
                                                    image_width=self.image_width,
                                                    image_encoding_dim=self.image_encoding_dim,
                                                    text_encoding_dim=self.text_encoding_dim,
                                                    hidden_dim=self.hidden_dim,
                                                    num_answers=self.num_answers)
        self.language = VisualReasoningLanguage(self.image_features, self.parameters)

    def test_get_nonterminal_productions(self):
        productions = self.language.get_nonterminal_productions()
        assert set(productions.keys()) == {
                '@start@',
                'Attention',
                'Answer',
                '<Attention:Answer>',
                '<Attention:Attention>',
                '<Attention,Attention:Answer>',
                '<Attention,Attention:Attention>',
                }
        check_productions_match(productions['@start@'],
                                ['Answer'])
        check_productions_match(productions['Attention'],
                                ['find',
                                 '[<Attention:Attention>, Attention]',
                                 '[<Attention,Attention:Attention>, Attention, Attention]'])
        check_productions_match(productions['Answer'],
                                ['[<Attention:Answer>, Attention]',
                                 '[<Attention,Attention:Answer>, Attention, Attention]'])
        check_productions_match(productions['<Attention:Answer>'],
                                ['exist', 'count', 'describe'])
        check_productions_match(productions['<Attention:Attention>'],
                                ['relocate', 'filter'])
        check_productions_match(productions['<Attention,Attention:Answer>'],
                                ['compare', 'count_equals', 'more', 'less'])
        check_productions_match(productions['<Attention,Attention:Attention>'],
                                ['and_', 'or_'])

    def test_find_returns_correct_shape(self):
        attended_question = torch.rand(self.text_encoding_dim)
        attention = self.language.find(attended_question)
        assert attention.size() == (self.image_height, self.image_width)

    def test_relocate_returns_correct_shape(self):
        attended_question = torch.rand(self.text_encoding_dim)
        attention = torch.rand(self.image_height, self.image_width)
        new_attention = self.language.relocate(attention, attended_question)
        assert new_attention.size() == (self.image_height, self.image_width)

    def test_filter_returns_correct_shape(self):
        attended_question = torch.rand(self.text_encoding_dim)
        attention = torch.rand(self.image_height, self.image_width)
        new_attention = self.language.filter(attention, attended_question)
        assert new_attention.size() == (self.image_height, self.image_width)

    def test_and_returns_correct_shape(self):
        attention1 = torch.rand(self.image_height, self.image_width)
        attention2 = torch.rand(self.image_height, self.image_width)
        new_attention = self.language.and_(attention1, attention2)
        assert new_attention.size() == (self.image_height, self.image_width)

    def test_or_returns_correct_shape(self):
        attention1 = torch.rand(self.image_height, self.image_width)
        attention2 = torch.rand(self.image_height, self.image_width)
        new_attention = self.language.or_(attention1, attention2)
        assert new_attention.size() == (self.image_height, self.image_width)

    def test_exist_returns_correct_shape(self):
        attention = torch.rand(self.image_height, self.image_width)
        answer = self.language.exist(attention)
        assert answer.size() == (self.num_answers,)

    def test_count_returns_correct_shape(self):
        attention = torch.rand(self.image_height, self.image_width)
        answer = self.language.count(attention)
        assert answer.size() == (self.num_answers,)

    def test_describe_returns_correct_shape(self):
        attended_question = torch.rand(self.text_encoding_dim)
        attention = torch.rand(self.image_height, self.image_width)
        answer = self.language.describe(attention, attended_question)
        assert answer.size() == (self.num_answers,)

    def test_compare_returns_correct_shape(self):
        attended_question = torch.rand(self.text_encoding_dim)
        attention1 = torch.rand(self.image_height, self.image_width)
        attention2 = torch.rand(self.image_height, self.image_width)
        answer = self.language.compare(attention1, attention2, attended_question)
        assert answer.size() == (self.num_answers,)

    def test_count_equals_returns_correct_shape(self):
        attention1 = torch.rand(self.image_height, self.image_width)
        attention2 = torch.rand(self.image_height, self.image_width)
        answer = self.language.count_equals(attention1, attention2)
        assert answer.size() == (self.num_answers,)

    def test_more_returns_correct_shape(self):
        attention1 = torch.rand(self.image_height, self.image_width)
        attention2 = torch.rand(self.image_height, self.image_width)
        answer = self.language.more(attention1, attention2)
        assert answer.size() == (self.num_answers,)

    def test_less_returns_correct_shape(self):
        attention1 = torch.rand(self.image_height, self.image_width)
        attention2 = torch.rand(self.image_height, self.image_width)
        answer = self.language.less(attention1, attention2)
        assert answer.size() == (self.num_answers,)

    def test_execute_logical_forms(self):
        # This just tests that execution _succeeds_ - we're not going to bother checking the
        # computation performed by each function, because there are learned parameters in there.
        # We'll treat this as similar to a model test, just making sure the tensor operations work.
        attended_question = {'attended_question': torch.rand(self.text_encoding_dim)}

        # A simple one to start with: (exist find)
        action_sequence = ['@start@ -> Answer',
                           'Answer -> [<Attention:Answer>, Attention]',
                           '<Attention:Answer> -> exist',
                           'Attention -> find']
        self.language.execute_action_sequence(action_sequence, [attended_question] * len(action_sequence))

        # (describe (and_ find find))
        action_sequence = ['@start@ -> Answer',
                           'Answer -> [<Attention:Answer>, Attention]',
                           '<Attention:Answer> -> describe',
                           'Attention -> [<Attention,Attention:Attention>, Attention, Attention]',
                           '<Attention,Attention:Attention> -> and_',
                           'Attention -> find',
                           'Attention -> find']
        self.language.execute_action_sequence(action_sequence, [attended_question] * len(action_sequence))

        # (compare (relocate find) (filter (or_ find find)))
        action_sequence = ['@start@ -> Answer',
                           'Answer -> [<Attention,Attention:Answer>, Attention, Attention]',
                           '<Attention,Attention:Answer> -> compare',
                           'Attention -> [<Attention:Attention>, Attention]',
                           '<Attention:Attention> -> relocate',
                           'Attention -> find',
                           'Attention -> [<Attention:Attention>, Attention]',
                           '<Attention:Attention> -> filter',
                           'Attention -> [<Attention,Attention:Attention>, Attention, Attention]',
                           '<Attention,Attention:Attention> -> or_',
                           'Attention -> find',
                           'Attention -> find']
        self.language.execute_action_sequence(action_sequence, [attended_question] * len(action_sequence))
