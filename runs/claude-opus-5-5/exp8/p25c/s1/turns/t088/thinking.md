I'm checking the format of gen_12.jsonl, where each line holds a problem id, source, answer, and a list of samples. The G11 and G2 dictionaries are keyed by (pid, idx) and store fields like src, text, pred, correct, tokens, and cut. So the loader will go line by line, pair each sample with its index as the key, and I still need to confirm which fields each sample record actually has.

