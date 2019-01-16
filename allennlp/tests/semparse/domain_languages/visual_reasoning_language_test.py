import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.domain_languages import VisualReasoningLanguage
from allennlp.tests.semparse.domain_languages.domain_language_test import check_productions_match


class VisualReasoningLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        image_height = 4
        image_width = 5
        image_encoding_dim = 3
        text_encoding_dim = 6
        hidden_dim = 7
        num_answers = 8
        self.image_features = torch.rand(image_width, image_height, image_encoding_dim)
        self.parameters = VisualReasoningParameters(image_height=image_height,
                                                    image_width=image_width,
                                                    image_encoding_dim=image_encoding_dim,
                                                    text_encoding_dim=text_encoding_dim,
                                                    hidden_dim=hidden_dim,
                                                    num_answers=num_answers)
        self.text_encoding_dim = text_encoding_dim
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

    def test_execute_logical_forms(self):
        # TODO(matt): need to define question attention, then pass it to execute() method, and
        # rewrite that method to get this to work.
        pass
