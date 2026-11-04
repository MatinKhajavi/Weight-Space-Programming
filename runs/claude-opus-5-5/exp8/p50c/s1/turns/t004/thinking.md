I'm noticing the model has no separate lm_head.weight, so the output projection must be tied to the embeddings. There are also no vision tensors in the list, just language_model weights, 320 in total. Next I'll check the chat template and the tokenizer.

