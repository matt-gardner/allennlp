from typing import Dict, List

import torch
from torch import Tensor
from torch.nn.functional import log_softmax

from allennlp.nn import util
from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, predicate,
                                                                predicate_with_side_args)


# An Attention is a single tensor; we're giving this a type so that we can use it for constructing
# predicates.
class Attention(Tensor):
    pass


# An Answer is a distribution over answer options, which is a tensor.  Again, we're subclassing
# Tensor here to give a concrete type for our predicates to use.
class Answer(Tensor):
    pass


class VisualReasoningShapesParameters(torch.nn.Module):
    """
    This is the same as :class:`VisualReasoningParameters`, but containing only a subset of the
    functions.  It was easier and cleaner to just duplicate the code instead of trying to
    complicate the full language to allow for only using some of the functions.
    """
    def __init__(self,
                 image_height: int,
                 image_width: int,
                 image_encoding_dim: int,
                 text_encoding_dim: int,
                 hidden_dim: int,
                 num_answers: int,
                 included_functions: List[str] = None) -> None:
        super().__init__()
        if text_encoding_dim != hidden_dim:
            raise ConfigurationError("The current implementation of `find` requires that "
                                     "hidden_dim == text_encoding_dim.")
        self.image_height = image_height
        self.image_width = image_width
        self.image_encoding_dim = image_encoding_dim
        self.text_encoding_dim = text_encoding_dim
        self.hidden_dim = hidden_dim
        self.num_answers = num_answers

        # The paper calls these "convolutions", but if you look at the code, it just does a 1x1
        # convolution, which is a linear transformation on the last dimension.  I'm still calling
        # them "convs" here, to match the original paper, and because they are operating on
        # image-feature-shaped things, and we might change our mind later about how to parameterize
        # them.  All of these parameters try to match what you see in table 1 of the paper, where
        # we write "W" as "linear".
        self.find_conv1 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.find_conv2 = torch.nn.Linear(hidden_dim, 1)
        self.relocate_conv1 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.relocate_conv2 = torch.nn.Linear(hidden_dim, 1)
        self.relocate_linear1 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.relocate_linear2 = torch.nn.Linear(text_encoding_dim, hidden_dim)
        self.exist_linear = torch.nn.Linear(image_height * image_width, num_answers)


class VisualReasoningShapesLanguage(DomainLanguage):
    """
    This is the same as :class:`VisualReasoningLanguage`, but containing only a subset of the
    functions.  It was easier and cleaner to just duplicate the code instead of trying to
    complicate the full language to allow for only using some of the functions.
    """

    def __init__(self,
                 image_features: Tensor,
                 encoded_question: Tensor,
                 parameters: VisualReasoningShapesParameters) -> None:
        super().__init__(start_types={Answer})
        if image_features is not None:
            # Pytorch puts channels first, but for the operations in here it's more convenient to
            # have channels last, so we transpose.
            image_features = image_features.permute(1, 2, 0)
        self.image_features = image_features
        self.encoded_question = encoded_question
        self.parameters = parameters
        if parameters:
            # We need to be able to instantiate an empty Language object in a few places, mostly so
            # we can interact with the grammar.
            ones = image_features.new_ones(self.parameters.image_height, self.parameters.image_width)
            self.uniform_attention = ones / self.parameters.image_height / self.parameters.image_width

    @predicate_with_side_args(['question_attention'])
    def find(self, question_attention: Tensor) -> Attention:
        attended_question = util.weighted_sum(self.encoded_question, question_attention)
        conv1 = self.parameters.find_conv1
        conv2 = self.parameters.find_conv2
        return conv2(conv1(self.image_features) * attended_question).squeeze()

    @predicate_with_side_args(['question_attention'])
    def relocate(self, attention: Attention, question_attention: Tensor) -> Attention:
        attended_question = util.weighted_sum(self.encoded_question, question_attention)
        conv1 = self.parameters.relocate_conv1
        conv2 = self.parameters.relocate_conv2
        linear1 = self.parameters.relocate_linear1
        linear2 = self.parameters.relocate_linear2
        attended_image = (attention.unsqueeze(-1) * self.image_features).sum(dim=[0, 1])
        return conv2(conv1(self.image_features) * linear1(attended_image) *
                     linear2(attended_question)).squeeze()

    @predicate
    def and_(self, attention1: Attention, attention2: Attention) -> Attention:
        return torch.max(torch.stack([attention1, attention2], dim=0), dim=0)[0]

    @predicate
    def exist(self, attention: Attention) -> Answer:
        linear = self.parameters.exist_linear
        return log_softmax(linear(attention.view(-1)), dim=-1)
