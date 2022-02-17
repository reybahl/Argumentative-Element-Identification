#Import required libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os #Traversing files

from tqdm import tqdm #Checking progress

train_df = pd.read_csv("../dataset/train.csv") #Read the training csv file

# Tagging Function
# B in front of the label means it is the beginning of the discourse, I in front means it is inside the discourse
def create_ner_tokens():
    tokenized_text_file = open("text.txt", "w")
    labels_file = open("labels.txt", "w")

    train_dir = f"../dataset/train"

    for filename in tqdm(os.listdir(train_dir)): #Loop through all the files in the training directory
        file_path = os.path.join(train_dir, filename) #Get file path

        # checking if it is a txt file
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == ".txt":

            file_id = os.path.splitext(filename)[0] #Splitting file name with the name and extension
            file_df = train_df[train_df["id"] == file_id] #Selecting the part of the training dataframe that contains info about the current file
            
            with open (file_path) as f:
                file_text = f.read()
            
            file_discourse_text_list = list(file_df.discourse_text)
            file_discourse_type_list = list(file_df.discourse_type)
            file_discourse_index_list = list(file_df.predictionstring)

            token_list = []
            label_list = []

            for i, discourse_type in enumerate(file_discourse_type_list):
                file_discourse_indices = str(file_discourse_index_list[i]).split() #Getting discourse indices from the predictionstring column
                discourse_start = int(file_discourse_indices[0])
                discourse_end = int(file_discourse_indices[-1])

                
                if not((i == 0) or (discourse_start == int(str(file_discourse_index_list[i-1]).split()[-1]) + 1)): #Checking if a part of the text is not a discourse
                    no_discourse_start_index = int(str(file_discourse_index_list[i-1]).split()[-1]) + 1
                    no_discourse_end_index = int(str(file_discourse_index_list[i]).split()[-1])
                    no_discourse_text = file_text.split()[no_discourse_start_index:no_discourse_end_index]

                    for token in no_discourse_text:  

                        token_list.append(token)
    
                        # Assigning label O for outside
                        label_list.append("O")
                

                discourse_text_tokens = file_discourse_text_list[i].split() #Split into word tokens

                for token_idx, token in enumerate(discourse_text_tokens):

                    token_list.append(token)

                    if token_idx == 0: #Checking if it is the first element
                        label_list.append("B-" + discourse_type.replace(" ", "")) #B for beginning (eg. B-Claim)
                    else:
                        label_list.append("I-" + discourse_type.replace(" ", "")) #I for inside (eg. I-Claim)


            tokenized_text_file.write(" ".join(token_list) + "\n") #Separating each token with a space and each essay with a newline
            labels_file.write(" ".join(label_list) + "\n") #Separating each label with a space and each essay's labels with a newline

    tokenized_text_file.close()
    labels_file.close()

create_ner_tokens()