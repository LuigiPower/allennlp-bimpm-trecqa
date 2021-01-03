# How to run

## BimPM
Running the training and validation process requires a reference to the jsonnet file, the directory in which
to store the metrics and trained models for each epoch. the -–include-package argument is required to make
AllenNLP load custom models and dataset readers, which it will detect thanks to theregisterdecorator.

```
allennlp train -s trecqaoutdir --include-package allennlp-bimpm-trecqa bimpmtrecqa.jsonnet
allennlp train -s wikiqaoutdir --include-package allennlp-bimpm-trecqa bimpmwikiqa.jsonnet
```

## ESIM
Modify the training script with the correct paths for the training and test datasets then run it

```
python trecqatrain_esim.py
python wikiqatrain_esim.py
```

## ESIM

## 1 INTRODUCTION

## 1.1 AllenNLP

AllenNLP is a research library built on PyTorch that simplies development of deep-learning models for Natural
Language Processing tasks.
It is comprised of two packages:
- allennlp: Contains all that is necessary for the development of a new model and JSONNET support.
- allennlp-models: Which hold all pre-implemented models and corresponding jsonnet samples. AllenNLP has been developed mostly for UNIX platforms, 
as on Windows the installation of jsonnet and allennlp-models packages fail the main runtime still works, only the more advanced features of jsonnet are
disabled (they will be considered plain JSON) and the pre-implemented models won’t be available unless a manual
installation is performed.
Implementing and running a model requires at least the following components:
- Model: model to run. It contains the deep network specication as it would be written in PyTorch, by
overriding methods dened in the AllenNLP Model class.
Dataset ReaderAllenNLP needs to understand the dataset it’s going to use for training and/or validation.
ere are two ways to effectively execute a full training/validation run:
- Training script: Write a Python script which initializes all the previously described objects in addition to
loading the data (either from file or from an URL) and creates a Trainer instance which takes care of
running the model.
- JSONNET: Specify the model, dataset reader, trainer and all required parameters in a single JSONNET
file. is method requires no code if the Dataset reader and Model are already implemented, which makes
it easy to congure and run.

## 2 READING THE DATASET

## 2.1 Basic DatasetReader implementation

eDatasetReaderclass provided by AllenNLP has areadmethod which must be overridden for the reader to
work.

```
Listing 1. Dataset reader
def _read(self , file_path: str) -> Iterator[Instance ]:
with open(file_path , encoding="utf -8") as f:
... file reading code ...
# Whenever an instance has been fully read , build and return it
yield self.text_to_instance (...)
Each Instance is a Sample that will be used for training, so it is initialized by passing a dictionary with three
keys: apremise, anhypothesisand alabelwhich species whether this answer is correct for the question.
```
```
Listing 2. Dataset reader
def text_to_instance(self , question_tokens: List[Token],
answer_tokens: List[Token], correct: str = None) -> Instance:
question_field = TextField(question_tokens , self.token_indexers)
answer_field = TextField(answer_tokens , self.token_indexers)
fields = {"premise": question_field , "hypothesis": answer_field}
```
```
if correct is not None:
fields["label"] = LabelField(int(correct), skip_indexing=True)
```
```
return Instance(fields)
To be able to use a custom Dataset Reader in both a JSONNET file or a training script, the reader must be
registered with a name using an AllenNLP decorator:
```
```
Listing 3. Dataset reader
@DatasetReader.register("datasetreadername")
the dataset readers for TrecQA and WikiQA will be built by using thistexttoinstancemethod and by overriding
read.
```
## 2.2 TrecQA

TrecQA contains the estion and Answer pairs in the following XML format:

```
Listing 4. Dataset sample
<QApairs id='1.4'>
<question >
What ethnic group / race are Crip members?
WP JJ NN IN NN VBP JJ NNS.
VMOD NMOD NMOD NMOD PMOD ROOT NMOD PRD P
6 3 1 1 4 0 8 6 6
- - PER_DESC -B - PER_DESC -B - ORGANIZATION -B PER_DESC -B -
```

### Evaluating BIMPM and ESIM models implemented in the AllenNLP open source NLP research library • 1:

```
</question >
<negative >
`` If they respect us , we respect them , '' one Crip said.
`` IN PRP VBD PRP , PRP VB PRP , '' CD NNP VBD.
P VMOD SUB SBAR OBJ P SUB VMOD OBJ P P NMOD SUB ROOT P
14 8 4 2 4 8 8 14 8 14 14 13 14 0 14
- - - - - - - - - - - - PERSON -B - -
</negative >
</QApairs >
```

Each question is delimited by a QApairs element, inside which there will always only be onequestionand
multiple answers, which are either labelednegativeorpositive.
Using an XML parser we can extract these elements and construct the required instances:

```
Listing 5. Dataset reader
def _read(self , file_path: str) -> Iterator[Instance ]:
import xml.etree.ElementTree as ET
root = ET.parse(file_path ). getroot ()
for questionpair in root.findall('QApairs'):
question = questionpair.find('question')
qtxtlines = question.text.split('\n')
qtxt = qtxtlines [1]. replace('\t', '')
positive = questionpair.findall('positive')
negative = questionpair.findall('negative')
```
```
for pa in positive:
patxtlines = pa.text.split('\n')
patxt = patxtlines [1]. replace('\t', '')
```
```
yield self.text_to_instance ([ Token(word) for word in qtxt],
[Token(word) for word in patxt], ' 1 ')
for na in negative:
natxtlines = na.text.split('\n')
natxt = natxtlines [1]. replace('\t', '')
```
```
yield self.text_to_instance ([ Token(word) for word in qtxt],
[Token(word) for word in natxt], ' 0 ')
```
## 2.3 WikiQA

WikiQA contains the estion and Answer pairs in the following tsv format:

```
Listing 6. Dataset sample
QuestionID \\ tQuestion \\ tDocumentID \\ tDocumentTitle \\ tSentenceID \\ tSentence \\ tLabel
Q1 question D1 document_title D1 -0 answer 0
```

In this case the data is a simple TSV, in which the label is on the last column and assumes the value of ’1’ for
correct answers. Splitting each line by the tabulation character, we can extract all the required information:

```
Listing 7. Dataset reader
def _read(self , file_path: str) -> Iterator[Instance ]:
with open(file_path , encoding="utf -8") as f:
for line in f:
questionid , question , documentid ,
documenttitle , sentenceid , sentence , correct = line.strip (). split('\t')
if questionid == "QuestionID":
continue
yield self.text_to_instance ([ Token(word) for word in question],
[Token(word) for word in sentence], correct)
```
## 3 TRAINING

## 3.1 BimPM

BimPM was run by using a jsonnet file. Note the dataset reader type which contains the name of the previously
created dataset readers.

```
Bimpm jsonnet file
1 {
2 "dataset_reader": {
3 "type": "trecqa", // or "wikiqa"
4 "token_indexers": {
5 "tokens": {
6 "type": "single_id",
7 "lowercase_tokens": false
8 },
9 "token_characters": {
10 "type": "characters"
11 }
12 }
13 },
14 "train_data_path": "TRAIN.xml", // Full path
15 "validation_data_path": "TEST.xml", // Full path
16 "model": {
17 "type": "bimpm",
18 "dropout": 0.1,
19 "text_field_embedder": {
20 "token_embedders": {
21 "tokens": {
22 "type": "embedding",
```

### Evaluating BIMPM and ESIM models implemented in the AllenNLP open source NLP research library • 1:

```
23 "pretrained_file": "https:// allennlp.s3.amazonaws.com/
datasets/glove/glove.840B.300d.txt.gz",
24 "embedding_dim": 300,
25 "trainable": true
26 },
27 "token_characters": {
28 "type": "character_encoding",
29 "embedding": {
30 "embedding_dim": 20,
31 "padding_index": 0
32 },
33 "encoder": {
34 "type": "gru",
35 "input_size": 20,
36 "hidden_size": 50,
37 "num_layers": 1,
38 "bidirectional": true
39 }
40 }
41 }
42 },
43 "matcher_word": {
44 "is_forward": true,
45 "hidden_dim": 400,
46 "num_perspectives": 10,
47 "with_full_match": false
48 },
49 "encoder1": {
50 "type": "lstm",
51 "bidirectional": true,
52 "input_size": 400,
53 "hidden_size": 200,
54 "num_layers": 1
55 },
56 "matcher_forward1": {
57 "is_forward": true,
58 "hidden_dim": 200,
59 "num_perspectives": 10
60 },
61 "matcher_backward1": {
62 "is_forward": false,
63 "hidden_dim": 200,
64 "num_perspectives": 10
65 },
66 "encoder2": {
67 "type": "lstm",
```

```
68 "bidirectional": true,
69 "input_size": 400,
70 "hidden_size": 200,
71 "num_layers": 1
72 },
73 "matcher_forward2": {
74 "is_forward": true,
75 "hidden_dim": 200,
76 "num_perspectives": 10
77 },
78 "matcher_backward2": {
79 "is_forward": false,
80 "hidden_dim": 200,
81 "num_perspectives": 10
82 },
83 "aggregator":{
84 "type": "lstm",
85 "bidirectional": true,
86 "input_size": 264,
87 "hidden_size": 100,
88 "num_layers": 2,
89 "dropout": 0.
90 },
91 "classifier_feedforward": {
92 "input_dim": 400,
93 "num_layers": 2,
94 "hidden_dims": [200, 2],
95 "activations": ["relu", "linear"],
96 "dropout": [0.1, 0.0]
97 }
98 },
99 "data_loader": {
100 "batch_sampler": {
101 "type": "bucket",
102 "padding_noise": 0.1,
103 "sorting_keys": ["premise", "hypothesis"],
104 "batch_size": 32
105 }
106 },
107 "trainer": {
108 "num_epochs": 40,
109 "patience": 10,
110 "cuda_device": -1,
111 "grad_clipping": 5.0,
112 "validation_metric": "+ accuracy",
113 "optimizer": {
```

### Evaluating BIMPM and ESIM models implemented in the AllenNLP open source NLP research library • 1:

```
114 "type": "adam",
115 "lr": 0.
116 }
117 }
118 }

```

Running the training and validation process requires a reference to the jsonnet file, the directory in which
to store the metrics and trained models for each epoch. e–include-packageargument is required to make
AllenNLP load custom models and dataset readers, which it will detect thanks to theregisterdecorator.

```
Listing 9. Bimpm jsonnet file
allennlp train -s trecqaoutdir --include -package workdir bimpmtrecqa.jsonnet
allennlp train -s wikiqaoutdir --include -package workdir bimpmwikiqa.jsonnet
```
## 3.2 ESIM

ESIM was run by creating a dedicated python script.
the first step is reading the input les through the Dataset Reader. In this case the reader is loaded by directly
instantiating the required class, not by using it’s registered name.

```
Listing 10. Dataset initialization
reader = TrecQADatasetReader () # Or WikiQADatasetReader

train_dataset = reader.read(cached_path(path_to_training_file ))
validation_dataset = reader.read(cached_path(path_to_validation_file ))

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
the next step is the initialization of the Model itself. ESIM is the model taken from the allennlp-models library.
```

```
Listing 11. Model initialization
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM ,
HIDDEN_DIM ,
batch_first=True))
inference = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM ,
HIDDEN_DIM ,
batch_first=True))

encoder_dim = word_embeddings.get_output_dim ()

projection_feedforward = FeedForward(
encoder_dim * 4, 1, inference.get_input_dim (), Activation.by_name("elu")()
)

(batch_size , model_dim * 2 * 4)
output_feedforward = FeedForward(
lstm.get_output_dim () * 4, 1, 2, Activation.by_name("elu")()
)

output_logit = torch.nn.Linear(in_features =2,
out_features =2)

simfunc = BilinearAttention(encoder_dim , encoder_dim)

model = ESIM(vocab=vocab ,
text_field_embedder=word_embeddings ,
encoder=lstm ,
inference_encoder=inference ,
similarity_function=simfunc ,
projection_feedforward=projection_feedforward ,
output_feedforward=output_feedforward ,
output_logit=output_logit)
```

the final step is the creation of the Trainer instance, in this case a GradientDescentTrainer using the AdamOp-
timizer.

```
Listing 12. Training
def build_trainer(
model: Model ,
serialization_dir: str ,
train_loader: DataLoader ,
dev_loader: DataLoader
) -> Trainer:
parameters = [
[n, p]
for n, p in model.named_parameters () if p.requires_grad
]
optimizer = AdamOptimizer(parameters)
trainer = GradientDescentTrainer(
model=model ,
serialization_dir=serialization_dir ,
data_loader=train_loader ,
validation_data_loader=dev_loader ,
num_epochs =10000 ,
optimizer=optimizer ,
patience =
)
return trainer
```

### Evaluating BIMPM and ESIM models implemented in the AllenNLP open source NLP research library • 1:

the training loop is started by simply callingtrainon the resulting instance.builddataloadersis a function
that wraps the datasets in a PyTorchDataLoader, which takes care of shuing and managing the batch size.

```
Listing 13. Training
train_dataset.index_with(vocab)
validation_dataset.index_with(vocab)

train_loader , val_loader = build_data_loaders(train_dataset , validation_dataset)

trainer = build_trainer(
model ,
'outputfolder',
train_loader ,
val_loader
)

trainer.train ()
```

## 4 VALIDATION

```
All tests have been run until patience ran out, set at 10 epochs.
```
## 4.1 TrecQA

4.1.1 BimPM. Listing 14. Training
1 {
2 "best_epoch": 5,
3 "peak_worker_0_memory_MB": 0,
4 "peak_gpu_0_memory_MB": 5767,
5 "training_duration": "15 days, 4:34:43.202185",
6 "training_start_epoch": 0,
7 "training_epochs": 7,
8 "epoch": 7,
9 "training_accuracy": 0.9608364378381414,
10 "training_loss": 0.09648763911427741,
11 "training_worker_0_memory_MB": 0.0,
12 "training_gpu_0_memory_MB": 4540,
13 "validation_accuracy": 0.8714568226763348,
14 "validation_loss": 0.5668508663463095,
15 "best_validation_accuracy": 0.8787079762689519,
16 "best_validation_loss": 0.
17 }

4.1.2 ESIM. Listing 15. Training
1 {
2 "best_epoch": 18,
3 "best_validation_accuracy": 0.8127883981542519,
4 "best_validation_loss": 0.4539732775601901,
5 "peak_worker_0_memory_MB": 0,
6 "peak_gpu_0_memory_MB": 5320,
7 "training_duration": "0:45:52.445977",
8 "training_start_epoch": 18,
9 "training_epochs": 9,
10 "epoch": 27,
11 "training_accuracy": 0.8801317932493401,
12 "training_loss": 0.33595128491962695,
13 "training_worker_0_memory_MB": 0.0,
14 "training_gpu_0_memory_MB": 4800,
15 "validation_accuracy": 0.8127883981542519,
16 "validation_loss": 0.
17 }


```
Evaluating BIMPM and ESIM models implemented in the AllenNLP open source NLP research library • 1:
```
## 4.2 WikiQA

```
4.2.1 BimPM. Listing 16. Training
1 {
2 "best_epoch": 9,
3 "best_validation_accuracy": 0.9526358475263584,
4 "best_validation_loss": 0.17583578684092185,
5 "peak_cpu_memory_MB": 909.
6 }
```
4.2.2 ESIM. Listing 17. Training
1 {
2 "best_epoch": 65,
3 "peak_worker_0_memory_MB": 0,
4 "peak_gpu_0_memory_MB": 7047,
5 "training_duration": "2:06:11.648058",
6 "training_start_epoch": 0,
7 "training_epochs": 74,
8 "epoch": 74,
9 "training_accuracy": 0.9489194499017681,
10 "training_loss": 0.188258571556837,
11 "training_worker_0_memory_MB": 0.0,
12 "training_gpu_0_memory_MB": 7047,
13 "validation_accuracy": 0.9524736415247365,
14 "validation_loss": 0.1769457460005421,
15 "best_validation_accuracy": 0.9524736415247365,
16 "best_validation_loss": 0.
17 }

## 5 CONCLUSIONS

```
While on the WikiQA dataset both models perform similarly and with good accuracy, on TrecQA both yield
mixed results, with BIMPM having slightly less validation loss.
```

