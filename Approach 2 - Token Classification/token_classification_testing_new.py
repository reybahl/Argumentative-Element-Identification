import csv
import os

import nltk
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)


class ArgumentativeElementIdentifier:
    def __init__(self, saved_model):
        self.discourse_types = ["O", 
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
        self.batch_size = 1
        self.min_tokens = 5
        self.tok_checkpoint = saved_model
        self.model_checkpoint = os.path.join(saved_model, "pytorch_model.bin")

        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_checkpoint, add_prefix_space=True)

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(self.tok_checkpoint, num_labels=len(self.discourse_types))

        self.model.load_state_dict(torch.load(self.model_checkpoint, map_location=torch.device('cpu')))
        self.model.eval()


        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        self.trainer = Trainer(
            self.model,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )

    def tokenize(self, examples):

        tokenized_inputs = self.tokenizer(examples["Tokens"], truncation=True, is_split_into_words=True, padding=True, return_offsets_mapping=True)

        return tokenized_inputs
    
    def create_tokens(self, text):
        
        tokenized_text = ""


        token_list = []
                
        text_tokens = text.split() #Split into word tokens

        for token in text_tokens:
            token_list.append(token)

        tokenized_text += (" ".join(token_list) + "\n") #Separating each token with a space and each essay with a newline

        return tokenized_text

    def get_discourse_type(self, idx):
        discourse_type = self.discourse_types[int(idx)]
        if discourse_type != "O":
            discourse_type = discourse_type[2:] #Slice to remove IOB TAG
            if discourse_type == "ConcludingStatement":
                discourse_type = "Concluding Statement"
        
        return discourse_type


    def predict(self, text):
        
        tokenized_text = self.create_tokens(text)
        test_tokens_df = pd.DataFrame([x for x in tokenized_text.split("\n")], columns = ["Tokens"])
#         test_tokens_df.name = "Tokens"
        test_tokens_df.Tokens = test_tokens_df.Tokens.str.split()
        test_dataset = Dataset.from_pandas(test_tokens_df)
        tokenized_test = test_dataset.map(self.tokenize, batched=True)
        predictions, _, _ = self.trainer.predict(tokenized_test)
        preds = np.argmax(predictions, axis=-1)

        return self.get_pred_token_ids(preds, tokenized_test["input_ids"])
        

    def get_pred_token_ids(self, preds, token_ids):
    
        discourse_type_with_string_list = []

        previous_discourse_type = ""
        
        
        for text_idx, text_preds in enumerate(preds):
            
            current_discourse_token_idx_list = []

            for i, discourse_pred in tqdm(enumerate(text_preds)):
                try: 
                    token_id_list = token_ids[text_idx][i]
                except:
                    break
                    
                if "<" in self.tokenizer.decode(token_id_list):
                    continue
                
                discourse_type = self.get_discourse_type(discourse_pred)
                
                
                if discourse_type != previous_discourse_type and i > 0:
                    try:
                        if len(self.tokenizer.decode(current_discourse_token_idx_list).split()) > 5:
                            # new_idx = current_idx + len(tokenizer.decode(current_discourse_token_idx_list).split())

                            discourse_type_with_string_list.append({
                               previous_discourse_type : self.tokenizer.decode(current_discourse_token_idx_list)
                            })

#                             discourse_type_list.append(previous_discourse_type)

                        current_discourse_token_idx_list = []
                    except:
                        continue

                current_discourse_token_idx_list.append(int(token_ids[text_idx][i]))

                previous_discourse_type = discourse_type
                
        return discourse_type_with_string_list

if __name__ == "__main__":
    element_identifier = ArgumentativeElementIdentifier("../saved_model")
    input_text = input()

    print(element_identifier.predict(input_text))