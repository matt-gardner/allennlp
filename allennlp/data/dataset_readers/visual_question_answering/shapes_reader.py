from glob import glob
from typing import Dict
import json
import os

from overrides import overrides
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, IndexField, LabelField, ListField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.semparse.domain_languages.visual_reasoning_language import \
        VisualReasoningLanguage, convert_reverse_polish_to_logical_form


@DatasetReader.register("shapes")
class ShapesReader(DatasetReader):
    """

    NOTE: Because we need to read several files for each split, the ``read()`` method expects a
    `glob` as input, like ``"path/to/data/train.tiny*"``.  We will use the files matching this glob
    to construct our inputs (additionally looking for an ``image_mean.npy`` file in the same
    directory).

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for the utterance.  See :class:`Tokenizer`.  Default is
        ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for the utterance.  See :class:`TokenIndexer`.  Default is
        ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional
        Whether to load data lazily.  Passed to super class.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False):
        super().__init__(lazy)
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
        files = self._get_files(file_path)

        image_mean = numpy.load(files['image_mean_file'])
        images = numpy.load(files['image_file'])
        utterances = open(files['question_file']).readlines()
        answers = open(files['answer_file']).readlines()
        logical_forms = json.load(open(files['logical_form_file']))

        for i in range(len(utterances)):
            # I didn't see this mentioned in the paper anywhere, but if you look at the code for
            # the original N2NMN paper, the images for SHAPES are mean-shifted.
            yield self.text_to_instance(images[i] - image_mean,
                                        utterances[i].strip(),
                                        answers[i].strip(),
                                        convert_reverse_polish_to_logical_form(logical_forms[i]))

    @overrides
    def text_to_instance(self,  # type: ignore
                         image: numpy.ndarray,
                         utterance: str,
                         denotation: str = None,
                         logical_form: str = None) -> Instance:
        # For SHAPES, the "utterances" are all "questions" and all "denotations" are "answers", but
        # we're using "utterance" and "denotation" here because these names determine what names
        # the model uses, and the model is more general than just "questions".
        tokenized_utterance = self._tokenizer.tokenize(utterance.lower())
        utterance_field = TextField(tokenized_utterance, self._token_indexers)
        # We need the images to have shape (channels, height, width).
        fields = {'image': ArrayField(image.transpose(2, 0, 1)),
                  'utterance': utterance_field,
                  'actions': self._production_rule_field}

        if denotation:
            fields['denotation'] = LabelField(denotation)
        if logical_form:
            actions = self._language.logical_form_to_action_sequence(logical_form)
            index_fields = []
            for production_rule in actions:
                index_fields.append(IndexField(self._action_map[production_rule],
                                               self._production_rule_field))
            fields['target_action_sequence'] = ListField(index_fields)
        return Instance(fields)

    @staticmethod
    def _get_files(file_path: str) -> Dict[str, str]:
        files = {}
        for filename in glob(file_path):
            if filename.endswith('.input.npy'):
                files['image_file'] = filename
            elif filename.endswith('.query_str.txt'):
                files['question_file'] = filename
            elif filename.endswith('.output'):
                files['answer_file'] = filename
            elif filename.endswith('.query_layout_symbols.json'):
                files['logical_form_file'] = filename

        if files.keys() != {'image_file', 'question_file', 'answer_file', 'logical_form_file'}:
            raise ConfigurationError(f"Did not find the required files with glob {file_path} "
                                     f"(only found {files.keys()})")
        data_directory = files['image_file'].rsplit('/', 1)[0]
        files['image_mean_file'] = data_directory + '/image_mean.npy'
        if not os.path.exists(files['image_mean_file']):
            raise ConfigurationError(f"Did not find image mean file in {data_directory}")
        return files
