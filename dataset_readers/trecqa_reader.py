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

@DatasetReader.register("trecqa")
class TrecQADatasetReader(DatasetReader):
        """
        DatasetReader for Question answering data, one question -> answer pair per line:

            <QApairs id='1.4'>
                <question>
                What	ethnic	group	/	race	are	Crip	members	?
                WP	JJ	NN	IN	NN	VBP	JJ	NNS	.
                VMOD	NMOD	NMOD	NMOD	PMOD	ROOT	NMOD	PRD	P
                6	3	1	1	4	0	8	6	6
                -	-	PER_DESC-B	-	PER_DESC-B	-	ORGANIZATION-B	PER_DESC-B	-
                </question>
                <negative>
                Prison	gangs	have	a	de	facto	negotiation	system	to	defuse	potential	conflicts	,	black	gang	members	said	.
                NNP	NNS	VBP	DT	JJ	NN	NN	NN	TO	VB	JJ	NNS	,	JJ	NN	NNS	VBD	.
                NMOD	SUB	VMOD	NMOD	NMOD	NMOD	NMOD	OBJ	VMOD	NMOD	NMOD	OBJ	P	NMOD	NMOD	SUB	ROOT	P
                2	3	17	8	8	8	8	3	10	8	12	10	17	16	16	17	0	17
                -	PER_DESC-B	-	-	-	-	-	-	-	-	-	-	-	-	PER_DESC-B	PER_DESC-I	-	-
                </negative>
                <negative>
                Nor	does	it	count	many	street	gangs	,	whose	members	may	loosely	organize	behind	bars	.
                NNP	VBZ	PRP	VB	JJ	JJ	NNS	,	WP$	NNS	MD	VB	JJ	IN	NNS	.
                SUB	ROOT	SUB	VMOD	NMOD	NMOD	OBJ	P	NMOD	NMOD	SBAR	VC	VMOD	VMOD	PMOD	P
                2	0	4	2	7	7	4	7	7	9	9	11	12	12	14	2
                -	-	-	-	-	-	PER_DESC-B	-	-	PER_DESC-B	-	-	-	-	-	-
                </negative>
                <negative>
                ``	There	was	a	time	when	most	of	the	gang	members	in	Texas	prisons	were	gang	members	who	had	joined	in	prison	.
                ``	EX	VBD	DT	NN	WRB	JJS	IN	DT	NN	NNS	IN	NNP	NNS	VBD	JJ	NNS	WP	VBD	VBN	IN	NN	.
                P	SUB	ROOT	NMOD	PRD	NMOD	SUB	NMOD	NMOD	NMOD	SUB	NMOD	NMOD	PMOD	SBAR	NMOD	PRD	NMOD	SBAR	VC	VMOD	PMOD	P
                3	3	0	5	3	5	15	7	11	11	15	11	14	12	6	17	15	17	18	19	20	21	3
                -	-	-	-	-	-	-	-	-	PER_DESC-B	PER_DESC-I	-	GPE-B	FAC_DESC-B	-	PER_DESC-B	PER_DESC-I	-	-	-	-	FAC_DESC-B	-
                </negative>
                <negative>
                But	now	the	street	gang	members	who	are	being	incarcerated	actually	outnumber	what	we	were	historically	calling	prison	gang	members	.	''
                CC	RB	DT	JJ	NN	NNS	WP	VBP	VBG	VBN	RB	VBP	WP	PRP	VBD	RB	VBG	NN	NN	NNS	.	''
                VMOD	NMOD	NMOD	NMOD	NMOD	SUB	NMOD	SBAR	VC	VC	VMOD	ROOT	VMOD	SUB	SBAR	VMOD	VC	NMOD	NMOD	OBJ	P	P
                12	6	6	6	6	12	6	7	8	9	12	0	12	15	13	15	15	20	20	17	12	12
                -	-	-	-	PER_DESC-B	PER_DESC-I	-	-	-	-	-	-	-	-	-	-	-	-	PER_DESC-B	PER_DESC-I	-	-
                </negative>
                <negative>
                Crips	members	,	who	did	not	want	to	be	identified	,	note	that	even	white	supremacist	groups	like	the	Aryan	Brotherhood	generally	refrain	from	collective	attacks	on	blacks	.
                NNP	NNS	,	WP	VBD	RB	VB	TO	VB	VBN	,	NN	IN	RB	JJ	NN	NNS	IN	DT	NNP	NNPS	RB	VB	IN	JJ	NNS	IN	NNS	.
                NMOD	NMOD	P	NMOD	SBAR	VMOD	VC	VMOD	VMOD	VC	P	ROOT	NMOD	NMOD	NMOD	NMOD	SUB	NMOD	NMOD	NMOD	PMOD	VMOD	SBAR	VMOD	NMOD	PMOD	NMOD	PMOD	P
                2	12	2	2	4	5	5	9	7	9	2	0	12	17	17	17	23	17	21	21	18	23	13	23	26	24	26	27	12
                ORGANIZATION-B	PER_DESC-B	-	-	-	-	-	-	-	-	-	-	-	-	-	-	ORG_DESC-B	-	-	ORGANIZATION-B	ORGANIZATION-I	-	-	-	-	-	-	PER_DESC-B	-
                </negative>
                <negative>
                ``	If	they	respect	us	,	we	respect	them	,	''	one	Crip	said	.
                ``	IN	PRP	VBD	PRP	,	PRP	VB	PRP	,	''	CD	NNP	VBD	.
                P	VMOD	SUB	SBAR	OBJ	P	SUB	VMOD	OBJ	P	P	NMOD	SUB	ROOT	P
                14	8	4	2	4	8	8	14	8	14	14	13	14	0	14
                -	-	-	-	-	-	-	-	-	-	-	-	PERSON-B	-	-
                </negative>
                <negative>
                If	members	of	rival	gangs	tangle	over	a	personal	issue	,	in	theory	the	isolated	combatants	are	permitted	to	resolve	their	differences	without	gang	interference	.
                IN	NNS	IN	NN	NNS	NN	IN	DT	JJ	NN	,	IN	JJ	DT	JJ	NNS	VBP	VBN	TO	VB	PRP$	NNS	IN	NN	NN	.
                VMOD	SBAR	NMOD	NMOD	NMOD	PMOD	NMOD	NMOD	NMOD	PMOD	P	VMOD	NMOD	NMOD	NMOD	SUB	ROOT	VC	VMOD	VMOD	NMOD	OBJ	NMOD	NMOD	PMOD	P
                17	1	2	6	6	3	6	10	10	7	17	17	16	16	16	17	0	17	20	18	22	20	22	25	23	17
                -	PER_DESC-B	-	-	ORG_DESC-B	-	-	-	-	-	-	-	-	-	-	ORG_DESC-B	-	-	-	-	-	-	-	PER_DESC-B	-	-
                </negative>
                <negative>
                The	members	of	the	Crips	agree	they	would	die	for	their	gang	,	and	some	would	be	willing	to	kill	.
                DT	NNS	IN	DT	NNS	VBP	PRP	MD	VB	IN	PRP$	NN	,	CC	DT	MD	VB	JJ	TO	NN	.
                NMOD	SUB	NMOD	NMOD	PMOD	VMOD	SUB	VMOD	VC	VMOD	NMOD	PMOD	P	VMOD	SUB	ROOT	VC	PRD	AMOD	PMOD	P
                2	6	2	5	3	16	8	6	8	9	12	10	16	16	16	0	16	17	18	19	16
                -	PER_DESC-B	-	-	ORGANIZATION-B	-	-	-	-	-	-	PER_DESC-B	-	-	-	-	-	-	-	-	-
                </negative>
            </QApairs>

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
            import xml.etree.ElementTree as ET
            root = ET.parse(file_path).getroot()
            for questionpair in root.findall('QApairs'):
                question = questionpair.find('question')
                qtxtlines = question.text.split('\n')
                qtxt = qtxtlines[1].replace('\t', ' ')
                positive = questionpair.findall('positive')
                negative = questionpair.findall('negative')
                # print("Question: " + qtxt)
                for pa in positive:
                    patxtlines = pa.text.split('\n')
                    patxt = patxtlines[1].replace('\t', ' ')
                    # print("Positive: " + patxt)
                    yield self.text_to_instance([Token(word) for word in qtxt], [Token(word) for word in patxt], '1')
                for na in negative:
                    natxtlines = na.text.split('\n')
                    natxt = natxtlines[1].replace('\t', ' ')
                    # print("Negative: " + natxt)
                    yield self.text_to_instance([Token(word) for word in qtxt], [Token(word) for word in natxt], '0')

            # with open(file_path) as f:
            #     for line in f:
            #         questionid, question, documentid, documenttitle, sentenceid, sentence, correct = line.strip().split('\t')
            #         if questionid == "QuestionID":
            #             continue