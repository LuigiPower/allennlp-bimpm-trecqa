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

from allennlp.models.bimpm import BiMpm

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.similarity_functions import BilinearSimilarity, CosineSimilarity, DotProductSimilarity, LinearSimilarity, MultiHeadedSimilarity
from allennlp.modules.bimpm_matching import BimPmMatching
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

from allennlp.nn import Activation

from dataset_readers.wikiqa_reader import WikiQADatasetReader

torch.manual_seed(1)

reader = WikiQADatasetReader()

train_dataset = reader.read(cached_path(
    '/Users/federicogiuggioloni/Projects/qa-allennlp/WikiQACorpus/WikiQA-train.tsv'))
validation_dataset = reader.read(cached_path(
    '/Users/federicogiuggioloni/Projects/qa-allennlp/WikiQACorpus/WikiQA-test.tsv'))

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

"""
    Parameters
    ----------
    premise : Dict[str, torch.LongTensor]
        From a ``TextField``
    hypothesis : Dict[str, torch.LongTensor]
        From a ``TextField``
    label : torch.IntTensor, optional (default = None)
        From a ``LabelField``
    metadata : ``List[Dict[str, Any]]``, optional, (default = None)
        Metadata containing the original tokenization of the premise and
        hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
    Returns
    -------
    An output dictionary consisting of:
    label_logits : torch.FloatTensor
        A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
        probabilities of the entailment label.
    label_probs : torch.FloatTensor
        A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
        entailment label.
    loss : torch.FloatTensor, optional
        A scalar loss to be optimised.
"""

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

"""
    This ``Model`` implements the BiMpm sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    encoder : ``Seq2SeqEncoder``
        Used to encode the premise and hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between encoded
        words in the premise and words in the hypothesis.
    projection_feedforward : ``FeedForward``
        The feedforward network used to project down the encoded and enhanced premise and hypothesis.
    inference_encoder : ``Seq2SeqEncoder``
        Used to encode the projected premise and hypothesis for prediction.
    output_feedforward : ``FeedForward``
        Used to prepare the concatenated premise and hypothesis for prediction.
    output_logit : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
"""

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
inference = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
# esim = PytorchSeq2SeqWrapper(torch.nn.BiMpm(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

encoder_dim = word_embeddings.get_output_dim()

projection_feedforward = FeedForward(
            encoder_dim * 4, 1, inference.get_input_dim(), Activation.by_name("elu")()
        )

# (batch_size, model_dim * 2 * 4)
output_feedforward = FeedForward(
            lstm.get_output_dim() * 4, 1, 2, Activation.by_name("elu")()
        )

output_logit = torch.nn.Linear(in_features=2,
                                out_features=2)

simfunc = BilinearSimilarity(encoder_dim, encoder_dim)

# check_dimensions_match(
#             encoder_dim,
#             lstm.get_input_dim(),
#             "text field embedding dim",
#             "encoder input dim",
#         )
#         check_dimensions_match(
#             encoder.get_output_dim() * 4,
#             projection_feedforward.get_input_dim(),
#             "encoder output dim",
#             "projection feedforward input",
#         )
#         check_dimensions_match(
#             projection_feedforward.get_output_dim(),
#             inference_encoder.get_input_dim(),
#             "proj feedforward output dim",
#             "inference lstm input dim",
#         )

model = WikiQADatasetReader(vocab=vocab,
                text_field_embedder=word_embeddings,
                encoder=lstm,
                inference_encoder=inference,
                similarity_function=simfunc,
                projection_feedforward=projection_feedforward,
                output_feedforward=output_feedforward,
                output_logit=output_logit)

if torch.cuda.is_available():
    cuda_device = 0

    model = model.cuda(cuda_device)
else:

    cuda_device = -1

optimizer = optim.SGD(model.parameters(), lr=0.1)

iterator = BucketIterator(batch_size=2, sorting_keys=[("premise", "num_tokens")])

iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1,
                  cuda_device=cuda_device)

trainer.train()

# predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

# tag_logits = predictor.predict("The dog ate the apple")['tag_logits']

# tag_ids = np.argmax(tag_logits, axis=-1)

# print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

# Here's how to save the model.
with open("/tmp/wikiqamodel.th", 'wb') as f:
    torch.save(model.state_dict(), f)

vocab.save_to_files("/tmp/wikiqavucabulary")

# And here's how to reload the model.
# vocab2 = Vocabulary.from_files("/tmp/wikiqavucabulary")

# model2 = LstmTagger(word_embeddings, esim, vocab2)

# with open("/tmp/wikiqamodel.th", 'rb') as f:
#     model2.load_state_dict(torch.load(f))

# if cuda_device > -1:
#     model2.cuda(cuda_device)

# predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
# tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
# np.testing.assert_array_almost_equal(tag_logits2, tag_logits)

