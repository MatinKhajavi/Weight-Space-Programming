I'm going to check attention head 2 in layer 23, scoring each query position for how much it lands on the first token, itself, earlier identical tokens, and the induction pattern, split into the prompt-and-lead region and the loop region.

I'm loading tf_compact.json to read the loop structure, since the period token count, prompt length, and lead length may already be stored there, and I'm checking the signatures of rep_index and line_rep_profile to see whether they can compute the repetition index per token.

