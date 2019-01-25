from typing import Dict, List

from nltk import Tree
import torch
from torch import Tensor
from torch.nn import Parameter

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


class VisualReasoningParameters(torch.nn.Module):
    """
    Stores all of the parameters necessary for the various learned functions in the
    ``VisualReasoningLanguage``.  This needs to be constructed outside the ``Language`` object,
    because we create one of those objects per ``Instance``, while the parameters are shared across
    all ``Instances``.  This also needs to be a ``torch.nn.Module``, so that your ``Model`` can
    save it as a member variable and have its weights get updated.
    """
    def __init__(self,
                 image_height: int,
                 image_width: int,
                 image_encoding_dim: int,
                 text_encoding_dim: int,
                 hidden_dim: int,
                 num_answers: int) -> None:
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
        self.find_linear = torch.nn.Linear(text_encoding_dim, hidden_dim)
        self.find_same_as_linear1 = torch.nn.Linear(hidden_dim, 1)
        self.find_same_as_linear2 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.find_same_as_linear3 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.find_same_as_linear4 = torch.nn.Linear(text_encoding_dim, hidden_dim)
        self.relocate_conv1 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.relocate_conv2 = torch.nn.Linear(hidden_dim, 1)
        self.relocate_linear1 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.relocate_linear2 = torch.nn.Linear(text_encoding_dim, hidden_dim)
        self.exist_linear = torch.nn.Linear(image_height * image_width, num_answers)
        self.count_linear = torch.nn.Linear(image_height * image_width, num_answers)
        self.describe_linear1 = torch.nn.Linear(hidden_dim, num_answers)
        self.describe_linear2 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.describe_linear3 = torch.nn.Linear(text_encoding_dim, hidden_dim)
        self.count_equals_linear1 = torch.nn.Linear(image_height * image_width, num_answers)
        self.count_equals_linear2 = torch.nn.Linear(image_height * image_width, num_answers)
        self.more_linear1 = torch.nn.Linear(image_height * image_width, num_answers)
        self.more_linear2 = torch.nn.Linear(image_height * image_width, num_answers)
        self.less_linear1 = torch.nn.Linear(image_height * image_width, num_answers)
        self.less_linear2 = torch.nn.Linear(image_height * image_width, num_answers)
        self.compare_linear1 = torch.nn.Linear(hidden_dim, num_answers)
        self.compare_linear2 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.compare_linear3 = torch.nn.Linear(image_encoding_dim, hidden_dim)
        self.compare_linear4 = torch.nn.Linear(text_encoding_dim, hidden_dim)


class VisualReasoningLanguage(DomainLanguage):
    """
    Implements the visual question answering language from the `"End-to-End Module Networks" paper
    by Ronghang Hu, Jacob Andreas and colleagues<https://arxiv.org/abs/1704.05526>`_ .  This
    language has a `learned execution model`, meaning that the predicates in this language have
    learned parameters.

    Parameters
    ----------
    image_features : ``Tensor``
        A tensor of shape (image_height, image_width, image_encoding_dim), containing the image
        features at some layer of a ConvNet.
    parameters : ``VisualReasoningParameters``
        The learnable parameters that we should use when executing functions in this language.
    """

    def __init__(self, image_features: Tensor, parameters: VisualReasoningParameters) -> None:
        super().__init__(start_types={Answer})
        self.image_features = image_features
        self.parameters = parameters
        ones = image_features.new_ones(self.parameters.image_height, self.parameters.image_width)
        self.uniform_attention = ones / self.parameters.image_height / self.parameters.image_width

    @predicate
    def scene(self) -> Attention:
        """
        Returns a uniform attention.
        """
        return self.uniform_attention

    @predicate_with_side_args(['attended_question'])
    def find(self, attended_question: Tensor) -> Attention:
        conv1 = self.parameters.find_conv1
        conv2 = self.parameters.find_conv2
        return conv2(conv1(self.image_features) * attended_question).squeeze()

    @predicate_with_side_args(['attended_question'])
    def find_same_as(self, attention: Attention, attended_question: Tensor) -> Attention:
        linear1 = self.parameters.find_same_as_linear1
        linear2 = self.parameters.find_same_as_linear2
        linear3 = self.parameters.find_same_as_linear3
        linear4 = self.parameters.find_same_as_linear4
        attended_image = (attention.unsqueeze(-1) * self.image_features).sum(dim=[0, 1])
        return linear1(linear2(self.image_features) * linear3(attended_image) *
                       linear4(attended_question)).squeeze()

    @predicate_with_side_args(['attended_question'])
    def relocate(self, attention: Attention, attended_question: Tensor) -> Attention:
        conv1 = self.parameters.relocate_conv1
        conv2 = self.parameters.relocate_conv2
        linear1 = self.parameters.relocate_linear1
        linear2 = self.parameters.relocate_linear2
        attended_image = (attention.unsqueeze(-1) * self.image_features).sum(dim=[0, 1])
        return conv2(conv1(self.image_features) * linear1(attended_image) *
                     linear2(attended_question)).squeeze()

    @predicate_with_side_args(['attended_question'])
    def filter(self, attention: Attention, attended_question: Tensor) -> Attention:
        return self.and_(attention, self.find(attended_question))

    @predicate
    def and_(self, attention1: Attention, attention2: Attention) -> Attention:
        return torch.max(torch.stack([attention1, attention2], dim=0), dim=0)[0]

    @predicate
    def or_(self, attention1: Attention, attention2: Attention) -> Attention:
        return torch.min(torch.stack([attention1, attention2], dim=0), dim=0)[0]

    @predicate
    def exist(self, attention: Attention) -> Answer:
        linear = self.parameters.exist_linear
        return linear(attention.view(-1))

    @predicate
    def count(self, attention: Attention) -> Answer:
        linear = self.parameters.count_linear
        return linear(attention.view(-1))

    @predicate_with_side_args(['attended_question'])
    def describe(self, attention: Attention, attended_question: Tensor) -> Answer:
        linear1 = self.parameters.describe_linear1
        linear2 = self.parameters.describe_linear2
        linear3 = self.parameters.describe_linear3
        attended_image = (attention.unsqueeze(-1) * self.image_features).sum(dim=[0, 1])
        return linear1(linear2(attended_image) * linear3(attended_question))

    @predicate_with_side_args(['attended_question'])
    def compare(self, attention1: Attention, attention2: Attention, attended_question: Tensor) -> Answer:
        linear1 = self.parameters.compare_linear1
        linear2 = self.parameters.compare_linear2
        linear3 = self.parameters.compare_linear3
        linear4 = self.parameters.compare_linear4
        attended_image1 = (attention1.unsqueeze(-1) * self.image_features).sum(dim=[0, 1])
        attended_image2 = (attention2.unsqueeze(-1) * self.image_features).sum(dim=[0, 1])
        return linear1(linear2(attended_image1) * linear3(attended_image2) * linear4(attended_question))

    @predicate
    def count_equals(self, attention1: Attention, attention2: Attention) -> Answer:
        linear1 = self.parameters.count_equals_linear1
        linear2 = self.parameters.count_equals_linear2
        return linear1(attention1.view(-1)) + linear2(attention2.view(-1))

    @predicate
    def more(self, attention1: Attention, attention2: Attention) -> Answer:
        linear1 = self.parameters.more_linear1
        linear2 = self.parameters.more_linear2
        return linear1(attention1.view(-1)) + linear2(attention2.view(-1))

    @predicate
    def less(self, attention1: Attention, attention2: Attention) -> Answer:
        linear1 = self.parameters.less_linear1
        linear2 = self.parameters.less_linear2
        return linear1(attention1.view(-1)) + linear2(attention2.view(-1))


def convert_reverse_polish_to_logical_form(predicates: List[str]) -> str:
    """
    The End-to-End Module Networks datasets come with reverse polish annotations, with particular
    function names.  This function converts those into logical forms that we can use.
    """
    stack = []
    for predicate in predicates:
        if predicate == '_Find':
            stack.append(Tree('find', []))
        elif predicate == '_Scene':
            stack.append(Tree('scene', []))
        elif predicate == '_Transform':
            argument = stack.pop()
            stack.append(Tree('relocate', [argument]))
        elif predicate == '_Describe':
            argument = stack.pop()
            stack.append(Tree('describe', [argument]))
        elif predicate == '_Filter':
            argument = stack.pop()
            stack.append(Tree('filter', [argument]))
        elif predicate == '_And':
            argument1 = stack.pop()
            argument2 = stack.pop()
            stack.append(Tree('and_', [argument1, argument2]))
        elif predicate == '_Or':
            argument1 = stack.pop()
            argument2 = stack.pop()
            stack.append(Tree('or_', [argument1, argument2]))
        elif predicate == '_Answer':
            argument = stack.pop()
            stack.append(Tree('exist', [argument]))
        elif predicate == '_Exist':
            argument = stack.pop()
            stack.append(Tree('exist', [argument]))
        elif predicate == '_Count':
            argument = stack.pop()
            stack.append(Tree('count', [argument]))
        elif predicate == '_FindSameProperty':
            argument = stack.pop()
            stack.append(Tree('find_same_as', [argument]))
        elif predicate == '_SameProperty':
            argument1 = stack.pop()
            argument2 = stack.pop()
            stack.append(Tree('compare', [argument1, argument2]))
        elif predicate == '_EqualNum':
            argument1 = stack.pop()
            argument2 = stack.pop()
            stack.append(Tree('count_equals', [argument1, argument2]))
        elif predicate == '_MoreNum':
            argument1 = stack.pop()
            argument2 = stack.pop()
            stack.append(Tree('more', [argument1, argument2]))
        elif predicate == '_LessNum':
            argument1 = stack.pop()
            argument2 = stack.pop()
            stack.append(Tree('less', [argument1, argument2]))
        else:
            raise RuntimeError(f"Unrecognized predicate: {predicate}")
    assert len(stack) == 1
    return _tree_to_logical_form(stack[0])


def _tree_to_logical_form(tree: Tree) -> str:
    if len(tree) == 0:
        return tree.label()
    else:
        arguments = [_tree_to_logical_form(child) for child in tree]
        return f"({tree.label()} {' '.join(arguments)})"
