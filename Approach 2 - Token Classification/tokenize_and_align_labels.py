def tokenize_and_align_labels(examples, tokenizer, discourse_types):
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