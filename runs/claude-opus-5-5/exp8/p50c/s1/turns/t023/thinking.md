I'm computing per-sample loop info: whether each sample is cut, and where the loop starts in tokens. I can get exact token positions by tokenizing each text with the tokenizer, since 512 texts of about 100k characters is manageable.

I'm working out the token index of each sample's loop onset by encoding the text up to that character offset, and then the fraction of the first 8192 tokens that fall inside the loop. After that I'll correlate the averaged MLP activations with that fraction across the cut samples.

