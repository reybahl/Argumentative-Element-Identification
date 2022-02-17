from tqdm import tqdm
import os

def create_tokens(test_dir, output_file):
    tokenized_text_file = open(output_file, "w")

    for filename in tqdm(os.listdir(test_dir)): #Loop through all the files in the training directory
        file_path = os.path.join(test_dir, filename) #Get file path

        # checking if it is a txt file
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == ".txt":
            
            with open (file_path) as f:
                file_text = f.read()

            token_list = []
            
            text_tokens = file_text.split() #Split into word tokens

            for token in text_tokens:
                token_list.append(token)

            tokenized_text_file.write(" ".join(token_list) + "\n") #Separating each token with a space and each essay with a newline

    tokenized_text_file.close()
    return token_list