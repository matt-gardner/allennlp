from typing import Dict
import json

from overrides import overrides
import numpy

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, IndexField, LabelField, ListField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.semparse.domain_languages.visual_reasoning_language import \
        VisualReasoningLanguage, convert_reverse_polish_to_logical_form


class ShapesReader(DatasetReader):
    """

    Parameters
    ----------
    data_split : ``str``
        The shapes data has multiple files in a particular layout in a given directory.  We need to
        take the directory as our "path" in ``read()``, so this tells us which split we should read
        from in that directory (e.g., ``train.large``, ``train.tiny``, ``val``, ``test``).  Note
        that this means that you `must` have separate train and validation dataset readers, with
        different values of this parameter.
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional
        Whether to load data lazily.  Passed to super class.
    """
    def __init__(self,
                 data_split: str,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False):
        super().__init__(lazy)
        self._data_split = data_split
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._language = VisualReasoningLanguage(None, None)
        self._production_rules = self._language.all_possible_productions()
        self._action_map = {rule: i for i, rule in enumerate(self._production_rules)}
        production_rule_fields = [ProductionRuleField(rule, is_global_rule=True)
                                  for rule in self._production_rules]
        self._production_rule_field = ListField(production_rule_fields)

    @overrides
    def _read(self, file_path: str):
        if not file_path.endswith('/'):
            file_path += '/'
        image_file = f"{file_path}{self._data_split}.input.npy"
        question_file = f"{file_path}{self._data_split}.query_str.txt"
        answer_file = f"{file_path}{self._data_split}.output"
        logical_form_file = f"{file_path}{self._data_split}.query_layout_symbols.json"

        images = numpy.load(image_file)
        questions = open(question_file).readlines()
        answers = open(answer_file).readlines()
        logical_forms = json.load(open(logical_form_file))

        for i in range(len(questions)):
            yield self.text_to_instance(images[i],
                                        questions[i].strip(),
                                        answers[i].strip(),
                                        convert_reverse_polish_to_logical_form(logical_forms[i]))

    @overrides
    def text_to_instance(self,  # type: ignore
                         image: numpy.ndarray,
                         question: str,
                         answer: str = None,
                         logical_form: str = None) -> Instance:
        tokenized_question = self._tokenizer.tokenize(question.lower())
        question_field = TextField(tokenized_question, self._token_indexers)
        fields = {'image': ArrayField(image),
                  'question': question_field,
                  'actions': self._production_rule_field}

        if answer:
            fields['answer'] = LabelField(answer)
        if logical_form:
            actions = self._language.logical_form_to_action_sequence(logical_form)
            index_fields = []
            for production_rule in actions:
                index_fields.append(IndexField(self._action_map[production_rule],
                                               self._production_rule_field))
            fields['target_action_sequence'] = ListField(index_fields)
        return Instance(fields)
