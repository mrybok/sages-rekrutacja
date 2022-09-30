# sages-rekrutacja

---
Post-deadline code cleaning in this branch.

---
### Dependencies
`pip install requirements.txt`

or 

`conda create --name <env> --file requirements.txt`

or 

`conda env create -f environment.yml`

---

# Fake News Pipeline

`Fake_News_Preprocessing.ipynb` notebook in `notebooks/fake_news` prepares the data.
The results of the preprocessing are already saved in `preprocessed/fake_news/data/`, so there is no
need to rerun the notebook.

### Tokenization
```
from common.util import tokenize_texts

tokenize_texts(
    in_file='preprocessed/fake_news/data/text_dicts.pkl',
    out_file='preprocessed/fake_news/data/text_dicts.pkl',
    verbose=True
)
```
Tokenizes each text in the saved `text_dicts` to prepare it for BERT processing.
- `in_file` - path to pickled `text_dicts` dictionary with Headline & Body texts
- `out_file` - where to save updated `text_dicts` dictionary with `input_ids` fields populated
- `verbose` - whether to log tokenization progress with tqdm
```
# Required text_dicts format
text_dicts = {
    'head: {
        head_id: {'text': '...', 'input_ids': [...]},
        ...
    },
    'body': {
        body_id: {'text': '...', 'input_ids': [...]},
        ...
    }
    
}
```
Ready `text_dicts` dictionary after tokenization is in `preprocessed/fake_news/data/text_dicts.pkl`

### Embedding
```
from fake_news.util import get_embeddings

for split in ['train', 'valid', 'none', 'test']:
    get_embeddings(
        stances=f'preprocessed/fake_news/data/{split}.csv',
        text_dicts='preprocessed/fake_news/data/text_dicts.pkl',
        output_file=f'preprocessed/fake_news/embeddings/{split}.pkl',
        batch_size=1,
        gpu=True,
        verbose=True,
        save=True
    )
```
Feeds the tokenized Headline-Body pairs to BERT, to produce contextualized embeddings.
Extracts the output < CLS > token representations and saves it for classifier head usage. The output 
file holds a dictionary of format `{'X': embeddings, 'y': stance_targets}`, where `embeddings` and 
`stance_targets` are torch tensors. The above use case embeds all the dataset splits.
- `stances` - path to .csv file with following columns: Headline ID, Body ID, Stance. Stance 
field is optional. If the table has Stance column the embeddings will be saved together with `"y"` 
stance targets. Otherwise, only embeddings `"X"` are saved (`"y"` field is empty).
- `text_dicts` - path to pickled output of Tokenization as above
- `output_file` - path where the embeddings dictionary should be saved
- `batch_size` - feed the Headline-Body pairs into BERT in batches of given size
- `gpu` - whether to use CUDA to produce the embeddings
- `verbose` - whether to log embedding progress with tqdm
- `save` - whether to save the embeddings, the function also outputs the dictionary
if one wants to use them instead of saving.

Embedding with BERT of all dataset splits took about 1 hour on my personal Nvidia RTX 2060.
Ready embeddings are saved in `preprocessed/fake_news/embeddings/`.

### Training
```
from fake_news.util import train_classifier_head

train_classifier_head(
    train_set='preprocessed/fake_news/embeddings/train.pkl',
    experiment_dir='results/run_train',
    valid_set='preprocessed/fake_news/embeddings/valid.pkl',
    batch_size=1024,
    epochs=100,
    gpu=True,
    seed=0,
    tensorboard=True
)
```
Trains a classifier head out of saved contextualized BERT embeddings. When the `valid_set` is 
provided, saves the classifier head with the least validation loss as 
`<experiment_dir>/classifier_best.pth`. Always saves the model after all training epochs in 
`<experiment_dir>/classifier_last.pth`. `<experiment_dir>/history.pkl` holds a dictionary with the 
history of performance metrics in the training - losses, accuracy, F-1 scores for train and 
validation sets.
- `train_set` - path to embeddings saved by `get_embeddings` as in a section above. 
The embeddings must have a dictionary format `{'X': embeddings, 'y': stance_targets}`, with values
being torch tensors.
- `experiment_dir` - path to directory where the results of training are saved
- `valid_set` - path to embeddings of a validation set, same format as `train_set`. Can be `None`.
- `batch_size` - feed the embeddings into classifier head in batches of given size
- `epochs` - how long to train for
- `gpu` - whether to use CUDA in training
- `seed` - seed for random number generator & reproducibility
- `tensorboard` - whether to log a training process with tensorboard

A trained classifier head is available in `models/fake_news/fake_news_classifier.pth`.

### Evaluation
```
from fake_news.util import evaluate_wrappes

loss, y_true, y_pred = evaluate_wrapper(
    model_path='models/fake_news/fake_news_classifier.pth', 
    evaluation_dataset='preprocessed/fake_news/embeddings/test.pkl', 
    batch_size=1024, 
    gpu=True
)
```
Loads a saved classifier head in `model_path` and uses it to make predictions over the data in 
`evaluation_dataset`. The data in `evaluation_dataset` must be a result of previous embedding.

- `model_path` - path to saved classifier head as resut of training above
- `evaluation_dataset` - path to the saved embeddings of evaluation set
- `batch_size` - feed the embeddings into classifier head in batches of given size
- `gpu` - whether to use CUDA in evaluaton

The `Fake_News_Train.ipynb` notebook in `notebooks/fake_news/` also demonstrates the more detailed 
training pipeline.

---
# Tweets Pipeline
`Tweets_Preprocessing.ipynb` notebook in `notebooks/tweets` prepares the data.
The results of the preprocessing are already saved in `preprocessed/tweets/data/`, so there is no
need to rerun the notebook.

### Tokenization
```
from common.util import tokenize_texts

tokenize_texts(
    in_file='preprocessed/fake_news/data/text_dicts.pkl',
    out_file='preprocessed/fake_news/data/text_dicts.pkl',
    verbose=True
)
```
Tokenizes each text in the saved `text_dicts` to prepare it for BERT processing.
- `in_file` - path to pickled `text_dicts` dictionary with Tweet texts
- `out_file` - where to save updated `text_dicts` dictionary with `input_ids` fields populated
- `verbose` - whether to log tokenization progress with tqdm
```
# Required text_dicts format
text_dicts = {
    'tweets: {
        tweet_id: {'text': '...', 'input_ids': [...]},
        ...
    }
    
}
```
Ready `text_dicts` dictionary after tokenization is in `preprocessed/tweets/data/text_dicts.pkl`

### Embeddings
```
from tweets.util import get_embeddings

for split in ['train', 'valid', 'test']:
    get_embeddings(
        tweets=f'preprocessed/tweets/data/{split}.csv',
        text_dicts='preprocessed/tweets/data/text_dicts.pkl',
        output_file=f'preprocessed/tweets/embeddings/{split}.pkl',
        batch_size=1,
        gpu=True,
        verbose=True,
        save=True
    )
```
Feeds the tokenized Tweets to BERT, to produce contextualized embeddings.
Extracts the output < CLS > token representations, concatenates the time encodings with BERT 
embeddings. Saves the embeddings for classifier head usage. The output 
file holds a dictionary of format `{'X': embeddings, 'y': polarity_targets}`, where `embeddings` and 
`polarity_targets` are torch tensors. The above use case embeds all the dataset splits.
- `tweets` - path to .csv file with following columns: `'target', 'id', 'sin day', 'cos day', 
'sin hour', 'cos hour'`. `'target'` field is optional. If the table has `'target'` column the 
embeddings will be saved together with `"y"` polarity targets. Otherwise, only embeddings `"X"` are 
saved (`"y"` field is empty).
- all other inputs as in Fake News pipeline

Ready embeddings are saved in `preprocessed/tweets/embeddings/`.

The training and evaluation parts are as in Fake News pipeline.
The only difference is that the tweets function also accept `use_time_rep` flag, which determines
whether the classifier head uses the tweet time encodings during training / evaluation. The flag
also determines the dimensionality of the classifier head input. When `use_time_rep` is True the
loaded classifier head must have 772 input dimensions, otherwise 768.

A trained classifier head using the time encodings is available in 
`models/tweets/tweets_classifier.pth`.

The `Tweets_Train.ipynb` notebook in `notebooks/tweets/` also demonstrates the more detailed 
training pipeline.

---
## TO DO:
- Add function doc-strings & signatures