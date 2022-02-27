## Overview
Given an essay, this project uses machine learning and token classification to separate the essay into distinct argumentative elements (eg. Claim, Evidence, Counterclaim, etc.)

This is a token classification (NER) approach. A longformer (Long-Document transformer) is trained for this task using the HuggingFace Python library.

## Dataset
The dataset can be downloaded from Kaggle - https://www.kaggle.com/c/feedback-prize-2021/data. Save the dataset in a directory called "dataset". The code accesses the data in that directory.

## Running the Code
 - Install the required Python packages using `pip install -r requirements.txt`
 - Create tokens and labels by running `create_tokens_training.py`. This generates two files - text.txt and labels.txt. These will be used to train the model.
 - Train the model (GPU recommended) using the `token_classification_training.py`. Model will be saved in "/saved_model" directory.
 - Use the trained model to predict on new data by running `argumentative_element_identifier.py`

## Training Time
With a GPU (recommended), the model takes about 5 hours to run. Without a GPU, it takes much longer. 