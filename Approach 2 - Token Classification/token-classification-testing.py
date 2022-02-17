import csv
import os

import nltk
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)
from create_tokens import create_tokens

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["Tokens"], truncation=True, is_split_into_words=True)

    return tokenized_inputs


discourse_types = ["O", 
    "B-Position",
    "I-Position",
    "B-Evidence",
    "I-Evidence",
    "B-Counterclaim", 
    "I-Counterclaim", 
    "B-Rebuttal", 
    "I-Rebuttal", 
    "B-Claim", 
    "I-Claim", 
    "B-ConcludingStatement", 
    "I-ConcludingStatement",
    "B-Lead",
    "I-Lead",   
]

# Config
batch_size = 1
min_tokens = 5
tok_checkpoint = '../saved_model'
model_checkpoint = '../saved_model/pytorch_model.bin'

tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint, add_prefix_space=True)

# Load model
model = AutoModelForTokenClassification.from_pretrained(tok_checkpoint, num_labels=len(discourse_types))

model.load_state_dict(torch.load(model_checkpoint))
model.eval()


data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    data_collator=data_collator,
    tokenizer=tokenizer
)

TEST_DIR = f"../test"

token_list = create_tokens(TEST_DIR, "test_text.txt")

test_tokens_df = pd.read_csv("test_text.txt", sep="\n", header= None, names=["Tokens"], quoting= csv.QUOTE_NONE)
test_tokens_df.Tokens = test_tokens_df.Tokens.str.split()

test_dataset = Dataset.from_pandas(test_tokens_df)

tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test

predictions, _, _ = trainer.predict(tokenized_test)

preds = np.argmax(predictions, axis=-1)

def get_discourse_type(idx):
    discourse_type = discourse_types[int(idx)]
    if discourse_type != "O":
        discourse_type = discourse_type[2:] #Slice to remove IOB TAG
        if discourse_type == "ConcludingStatement":
            discourse_type = "Concluding Statement"
    
    return discourse_type


predictionstring_list = []
discourse_type_list = []
text_id_list = []

previous_discourse_type = ""
file_list = os.listdir(TEST_DIR)

for text_idx, text_preds in enumerate(preds):
    current_discourse_idx_list = []
    for i, discourse_pred in enumerate(text_preds):
        discourse_type = get_discourse_type(discourse_pred)
        
        if discourse_type != previous_discourse_type and i > 0:
            if len(current_discourse_idx_list) > 5:
                text_id_list.append(os.path.splitext(file_list[text_idx])[0])
                predictionstring_list.append(" ".join(current_discourse_idx_list))
                discourse_type_list.append(previous_discourse_type)
            
            current_discourse_idx_list = []
        
        current_discourse_idx_list.append(str(i))
        
        previous_discourse_type = discourse_type

fixed_id_list = []
fixed_discourse_type_list = []
fixed_predictionstring_list = []

for i, x in enumerate(discourse_type_list):
    
    if (text_id_list[i] == text_id_list[i-1]) and (discourse_type_list[i] == discourse_type_list[i-1]):
        fixed_predictionstring_list[-1] = fixed_predictionstring_list[-1] + predictionstring_list[i]
    else:
        fixed_id_list.append(text_id_list[i])
        fixed_discourse_type_list.append(discourse_type_list[i])
        fixed_predictionstring_list.append(predictionstring_list[i])
        

preds_df = pd.DataFrame({
    "id" : fixed_id_list,
    "class" : fixed_discourse_type_list,
    "predictionstring" : fixed_predictionstring_list
})


preds_df.to_csv("predictions.csv", index=False)
