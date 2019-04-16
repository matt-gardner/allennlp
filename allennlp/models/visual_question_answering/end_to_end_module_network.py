import logging
from typing import Any, Dict, List, Tuple

import difflib
import sqlparse
from overrides import overrides
import torch


from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.semparse.executors import SqlExecutor
from allennlp.models.model import Model
from allennlp.modules import Attention, ImageEncoder, Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.nn import util
from allennlp.semparse.domain_languages.visual_reasoning_language import (VisualReasoningLanguage,
                                                                          VisualReasoningParameters)
from allennlp.semparse.contexts.atis_sql_table_context import NUMERIC_NONTERMINALS
from allennlp.semparse.contexts.sql_context_utils import action_sequence_to_sql
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.transition_functions.basic_transition_function import BasicTransitionFunction
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.states import GrammarStatelet, RnnStatelet
from allennlp.training.metrics import Average
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("end_to_end_module_network")
class EndToEndModuleNetwork(Model):
    """
    A re-implementation of `End-to-End Module Networks for Visual Question Answering
    <https://www.semanticscholar.org/paper/Learning-to-Reason%3A-End-to-End-Module-Networks-for-Hu-Andreas/5e07d6951b7bc0c4113313a9586ce8178eacdf57>`_

    This implementation is based on our semantic parsing framework, and uses marginal likelihood to
    train the parser when labeled action sequences are not available.  It is `not` an exact
    re-implementation, but rather a very similar model with some significant differences in how the
    grammar is used.

    Parameters
    ----------
    vocab : ``Vocabulary``
    utterance_embedder : ``TextFieldEmbedder``
        Embedder for utterances.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input utterance.
    decoder_beam_search : ``BeamSearch``
        Beam search used to retrieve best sequences after training.
    max_decoding_steps : ``int``
        When we're decoding with a beam search, what's the maximum number of steps we should take?
        This only applies at evaluation time, not during training.
    attention: ``Attention``
        We compute an attention over the input utterance at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the transition function.
    image_shape : ``Tuple[int, int, int]``
        This is the shape that the input image representation will have, as ``(num_channels,
        height, width)``.  We need this because we need to know the final image feature
        representation dimension so that we can construct parameters in our program executor, which
        operates on that feature representation.  This means that you cannot change image sizes
        after training your model.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, we will learn a bias weight for each action that gets used when predicting
        that action, in addition to its embedding.
    dropout : ``float``, optional (default=0)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    use_gold_program_for_eval : ``bool``, optional (default=False)
        If true, we will use the gold program for evaluation when it is available (this only tests
        the program executor, not the parser).
    """
    def __init__(self,
                 vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 image_encoder: ImageEncoder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 attention: Attention,
                 image_shape: Tuple[int, int, int],
                 add_action_bias: bool = True,
                 decoder_num_layers: int = 1,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels',
                 denotation_namespace: str = 'labels',
                 use_gold_program_for_eval: bool = False) -> None:
        # Atis semantic parser init
        super().__init__(vocab)
        self._utterance_embedder = utterance_embedder
        self._image_encoder = image_encoder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._add_action_bias = add_action_bias
        self._dropout = torch.nn.Dropout(p=dropout)
        self._rule_namespace = rule_namespace
        self._denotation_namespace = denotation_namespace
        self._denotation_accuracy = denotation_namespace
        self._use_gold_program_for_eval = use_gold_program_for_eval

        self._denotation_accuracy = CategoricalAccuracy()
        # TODO(mattg): use FullSequenceMatch instead of this.
        self._program_accuracy = Average()

        self._action_padding_index = -1  # the padding value used by IndexField
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)

        image_feature_shape = image_encoder.get_output_shape_for_input(image_shape)
        image_encoding_dim, image_height, image_width = image_feature_shape
        self._language_parameters = VisualReasoningParameters(
                image_height=image_height,
                image_width=image_width,
                image_encoding_dim=image_encoding_dim,
                text_encoding_dim=self._encoder.get_output_dim(),
                hidden_dim=self._encoder.get_output_dim(),
                num_answers=vocab.get_vocab_size(self._denotation_namespace),
                )

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous utterance attention.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(torch.FloatTensor(encoder.get_output_dim()))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)

        self._decoder_num_layers = decoder_num_layers

        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood()
        self._transition_function = BasicTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                            action_embedding_dim=action_embedding_dim,
                                                            input_attention=attention,
                                                            predict_start_type_separately=False,
                                                            add_action_bias=self._add_action_bias,
                                                            dropout=dropout,
                                                            num_layers=self._decoder_num_layers)

        # Our language is constant across instances, so we just create one up front that we can
        # re-use to construct the `GrammarStatelet`.
        self._world = VisualReasoningLanguage(None, None)

    @overrides
    def forward(self,  # type: ignore
                image: torch.Tensor,
                utterance: Dict[str, torch.LongTensor],
                actions: List[ProductionRule],
                target_action_sequence: torch.LongTensor = None,
                denotation: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        In here, we set up the initial state for the decoder, then perform a search over programs,
        then execute those programs to get a denotation.  Both the search over programs and the
        program execution have parameters.  If we have supervision for the program (in
        ``target_action_sequence``) we do not perform search, but just compute a loss for this
        directly.  Either way, during training, we assume a correct denotation is given and we
        train on that signal.

        Parameters
        ----------
        image : torch.Tensor
            The image that corresponds to each instance, assumed to be output of an ``ArrayField``.
        utterance : Dict[str, torch.LongTensor]
            The output of ``TextField.as_array()`` applied on the utterance ``TextField``. This will
            be passed through a ``TextFieldEmbedder`` and then through an encoder.
        actions : ``List[ProductionRule]``
            A list of all possible actions available in the grammar (which is the same for each
            instance), indexed into a ``ProductionRule`` using a ``ProductionRuleField``.  We will
            embed all of these and use the embeddings to determine which action to take at each
            timestep in the decoder.
        target_action_sequence : torch.Tensor, optional (default=None)
            The action sequence for the correct action sequence, where each action is an index into
            the list of possible actions.  This tensor has shape ``(batch_size, sequence_length,
            1)``. We remove the trailing dimension.
        denotation : torch.Tensor, optional (default=None)
            The denotation of the utterance, given as a label to be used with cross entropy (this
            is pretty limiting, assuming that all possible denotations are known a priori, but it's
            what we do for now).
        """
        image_features = self._image_encoder(image)
        initial_state = self._get_initial_state(utterance, actions)
        batch_size = image.shape[0]
        if target_action_sequence is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequence = target_action_sequence.squeeze(-1)
            target_mask = target_action_sequence != self._action_padding_index
        else:
            target_mask = None

        if (self.training or self._use_gold_program_for_eval) and target_action_sequence is not None:
            # target_action_sequence is of shape (batch_size, 1, sequence_length) here after we
            # unsqueeze it for the MML trainer.
            search = ConstrainedBeamSearch(beam_size=None,
                                           allowed_sequences=target_action_sequence.unsqueeze(1),
                                           allowed_sequence_mask=target_mask.unsqueeze(1))
            final_states = search.search(initial_state, self._transition_function)
        else:
            final_states = self._beam_search.search(self._max_decoding_steps,
                                                    initial_state,
                                                    self._transition_function,
                                                    keep_final_unfinished_states=False)

        action_mapping = {}
        for action_index, action in enumerate(actions):
            action_mapping[action_index] = action[0]

        outputs: Dict[str, Any] = {'action_mapping': action_mapping}
        outputs['best_action_sequence'] = []
        outputs['debug_info'] = []

        losses = []
        for batch_index in range(batch_size):
            if not final_states[batch_index]:
                logger.error(f'No pogram found for batch index {batch_index}')
                outputs['best_action_sequence'].append([])
                outputs['debug_info'].append([])
                continue
            world = VisualReasoningLanguage(image_features[batch_index], self._language_parameters)
            denotation_log_prob_list = []
            # TODO(mattg): maybe we want to limit the number of states we evaluate (programs we
            # execute) at test time, just for efficiency.
            for state_index, state in enumerate(final_states[batch_index]):
                action_indices = state.action_history[0]
                action_strings = [action_mapping[action_index] for action_index in action_indices]
                question_attention = [info['question_attention'] for info in state.debug_info[0]]
                # Shape: (num_denotations,)
                state_denotation_log_probs = world.execute_action_sequence(action_strings, question_attention)
                # P(denotation | parse) * P(parse | question)
                denotation_log_prob_list.append(state_denotation_log_probs + state.score[0])
                if state_index == 0:
                    outputs['best_action_sequence'].append(action_strings)
                    outputs['debug_info'].append(state.debug_info[0])
                    if target_action_sequence:
                        targets = target_action_sequence[batch_index].data
                        program_correct = self._action_history_match(action_indices, targets)
                        self._program_accuracy(program_correct)

            # P(denotation | parse) * P(parse | question) for the all programs on the beam.
            # Shape: (beam_size, num_denotations)
            denotation_log_probs = torch.stack(denotation_log_prob_list)
            # \Sum_parse P(denotation | parse) * P(parse | question) = P(denotation | question)
            # Shape: (num_denotations,)
            marginalized_denotation_log_probs = util.logsumexp(denotation_log_probs, dim=0)
            if denotation:
                losses.append(-marginalized_denotation_log_probs[denotation[batch_index]])
                self._denotation_accuracy(marginalized_denotation_log_probs, denotation[batch_index])
        if losses:
            outputs['loss'] = torch.stack(losses).mean()
        return outputs

    def _get_initial_state(self,
                           utterance: Dict[str, torch.LongTensor],
                           actions: List[ProductionRule]) -> GrammarBasedState:
        embedded_utterance = self._utterance_embedder(utterance)
        utterance_mask = util.get_text_field_mask(utterance).float()

        batch_size = embedded_utterance.size(0)

        # (batch_size, num_utterance_tokens, embedding_dim)
        encoder_input = embedded_utterance

        # (batch_size, utterance_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, utterance_mask))

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             utterance_mask,
                                                             self._encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, self._encoder.get_output_dim())
        initial_score = embedded_utterance.data.new_zeros(batch_size)

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        utterance_mask_list = [utterance_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            if self._decoder_num_layers > 1:
                encoder_output = final_encoder_output[i].repeat(self._decoder_num_layers, 1)
                cell = memory_cell[i].repeat(self._decoder_num_layers, 1)
            else:
                encoder_output = final_encoder_output[i]
                cell = memory_cell[i]
            initial_rnn_state.append(RnnStatelet(encoder_output,
                                                 cell,
                                                 self._first_action_embedding,
                                                 self._first_attended_utterance,
                                                 encoder_output_list,
                                                 utterance_mask_list))


        initial_grammar_state = [self._create_grammar_state(actions) for _ in range(batch_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          possible_actions=actions,
                                          debug_info=[[] for _ in range(batch_size)])
        return initial_state

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
        # Check if target is big enough to cover prediction (including start/end symbols)
        if len(predicted) > targets.size(0):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        targets_trimmed = targets[:len(predicted)]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        return predicted_tensor.equal(targets_trimmed)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'denotation_acc': self._denotation_accuracy.get_metric(reset),
                'program_acc': self._program_accuracy.get_metric(reset),
                }

    def _create_grammar_state(self, possible_actions: List[ProductionRule]) -> GrammarStatelet:
        """
        This method creates the GrammarStatelet object that's used for decoding.  Part of creating
        that is creating the `valid_actions` dictionary, which contains embedded representations of
        all of the valid actions.  So, we create that here as well.

        The inputs to this method are for a `single instance in the batch`; none of the tensors we
        create here are batched.  We grab the global action ids from the input ``ProductionRules``,
        and we use those to embed the valid actions for every non-terminal type.

        Parameters
        ----------
        possible_actions : ``List[ProductionRule]``
            From the input to ``forward`` for a single batch instance.
        """
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = self._world.get_nonterminal_productions()

        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.

            action_indices = [action_map[action_string] for action_string in action_strings]
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]
            global_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                global_actions.append((production_rule_array[2], action_index))

            global_action_tensors, global_action_ids = zip(*global_actions)
            global_action_tensor = torch.cat(global_action_tensors, dim=0).long()
            global_input_embeddings = self._action_embedder(global_action_tensor)
            global_output_embeddings = self._output_action_embedder(global_action_tensor)
            translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                       global_output_embeddings,
                                                       list(global_action_ids))

        return GrammarStatelet([], translated_valid_actions, self._world.is_nonterminal)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in ``TransitionFunction``.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_actions`` to the ``output_dict``.
        """
        # TODO(mattg): FIX THIS - I haven't touched this method yet.
        action_mapping = output_dict['action_mapping']
        best_actions = output_dict["best_action_sequence"]
        debug_infos = output_dict['debug_info']
        batch_action_info = []
        for batch_index, (predicted_actions, debug_info) in enumerate(zip(best_actions, debug_infos)):
            instance_action_info = []
            for predicted_action, action_debug_info in zip(predicted_actions, debug_info):
                action_info = {}
                action_info['predicted_action'] = predicted_action
                considered_actions = action_debug_info['considered_actions']
                probabilities = action_debug_info['probabilities']
                actions = []
                for action, probability in zip(considered_actions, probabilities):
                    if action != -1:
                        actions.append((action_mapping[(batch_index, action)], probability))
                actions.sort()
                considered_actions, probabilities = zip(*actions)
                action_info['considered_actions'] = considered_actions
                action_info['action_probabilities'] = probabilities
                action_info['utterance_attention'] = action_debug_info.get('question_attention', [])
                instance_action_info.append(action_info)
            batch_action_info.append(instance_action_info)
        output_dict["predicted_actions"] = batch_action_info
        return output_dict
