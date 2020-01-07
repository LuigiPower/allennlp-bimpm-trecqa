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

@DatasetReader.register("wikiqa")
class WikiQADatasetReader(DatasetReader):
    """
    DatasetReader for Question answering data, one question -> answer pair per line:

        QuestionID\\tQuestion\\tDocumentID\\tDocumentTitle\\tSentenceID\\tSentence\\tLabel
        Q1  how are glacier caves formed?   D1  Glacier cave    D1-0    A partly submerged glacier cave on Perito Moreno Glacier .  0

    Label: 0 if wrong, 1 if correct
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, question_tokens: List[Token], answer_tokens: List[Token], correct: str = None) -> Instance:
        question_field = TextField(question_tokens, self.token_indexers)
        answer_field = TextField(answer_tokens, self.token_indexers)
        fields = {"premise": question_field, "hypothesis": answer_field}

        if correct is not None:
            # label_field = SequenceLabelField(labels=correct, sequence_field=sentence_field)
            fields["label"] = LabelField(int(correct), skip_indexing=True)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                questionid, question, documentid, documenttitle, sentenceid, sentence, correct = line.strip().split('\t')
                if questionid == "QuestionID":
                    continue
                yield self.text_to_instance([Token(word) for word in question], [Token(word) for word in sentence], correct)
