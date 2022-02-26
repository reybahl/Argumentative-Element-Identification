import csv
import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)


train_tokens_df = pd.read_csv("text.txt", sep="\n", header= None, names=["Tokens"], quoting=csv.QUOTE_NONE)
train_labels_df = pd.read_csv("labels.txt", sep="\n", header= None, names=["Labels"], quoting = csv.QUOTE_NONE)

train_df = pd.concat([train_tokens_df, train_labels_df], axis=1)
train_df.Tokens = train_df.Tokens.str.split()
train_df.Labels = train_df.Labels.str.split()

train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.train_test_split(test_size=0.1) #Train-val split


tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)

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

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["Tokens"], truncation=True, is_split_into_words=True, max_length=1024)

    labels = []
    for i, label in enumerate(examples["Labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(discourse_types.index(label[word_idx]))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

model = AutoModelForTokenClassification.from_pretrained("allenai/longformer-base-4096", num_labels=len(discourse_types))

os.environ["WANDB_DISABLED"] = "true"

BS = 4
GRAD_ACC = 8
LR = 5e-5
WD = 0.01
WARMUP = 0.1
N_EPOCHS = 5
OUTPUT_DIR = "./results"

training_args = TrainingArguments(
    evaluation_strategy = "epoch",
    logging_strategy = "epoch",
    save_strategy = "epoch",
    output_dir = OUTPUT_DIR,          
    learning_rate=LR,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    num_train_epochs=N_EPOCHS,
    weight_decay=WD,
    report_to=None, 
    gradient_accumulation_steps=GRAD_ACC,
    warmup_ratio=WARMUP
)

model = model.to("cuda:0")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train["train"],
    eval_dataset = tokenized_train["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

model.to("cuda")


print("training")
with tf.device('/GPU:0'):
    trainer.train()

trainer.save_model('../saved_model')
