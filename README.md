# sages-rekrutacja

---
## Dependencies
`pip install requirements.txt`

or 

`conda create --name <env> --file requirements.txt`

or 

`conda env create -f environment.yml`

---
# Files

`Fake_News_Exploration.ipynb`
- *"fake news"* dataset analysis and pre-processing
- discussions

Running the notebook pre-processes the dataset and prepares input for BERT transformer.
Input for BERT transformer is stored in `./preprocessed/fake_news/bert_loaders/<split_name>.pkl`

---

`python embeddings.py`

Feeds the outputs of the `Fake_News_Exploration.ipynb` notebook into BERT transformer and extracts 
the < CLS > token contextualized embedding for each example. Saves the embeddings for appropriate 
splits in `./preprocessed/fake_news/embeddings/<split_name>.pkl` files.

---

`Fake_News_Train.ipynb`
- tunes the early stopping hyper-parameter
- trains classifier
- saves the classifeir in `./models/fake_news_classifier.pth`
- evaluates performance on train & test set


---

## Classifier
The *"fake news"* classifier is a simple linear head, mapping the 768 dimensional BERT < CLS > token
embedding into 4 classes:

`model = torch.nn.Linear(768, 4)`