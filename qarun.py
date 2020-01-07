from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import LabelField, TextField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models.esim import ESIM

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.similarity_functions import BilinearSimilarity, CosineSimilarity, DotProductSimilarity, LinearSimilarity, MultiHeadedSimilarity
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

from allennlp.nn import Activation

torch.manual_seed(1)

if __name__ == '__main__':
    vocab2 = Vocabulary.from_files("./wikiqavucabulary")

    model2 = LstmTagger(word_embeddings, esim, vocab2)

    with open("./wikiqamodel.th", 'rb') as f:
        model2.load_state_dict(torch.load(f))

    if cuda_device > -1:
        model2.cuda(cuda_device)

    predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
    tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
    np.testing.assert_array_almost_equal(tag_logits2, tag_logits)
