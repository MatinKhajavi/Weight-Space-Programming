I'm noting that digits show the strongest relative changes, with the largest values in the L2 q_proj, followed by answer-frame and question words. The word " think" also stands out in the L1–L3 MLP gate, which makes me suspect the training data has many numeric and arithmetic questions along with "I think the answer is..." phrasing.

The training prompts likely use numeric answers and the phrase "I think," and the digit tokens are where the changes concentrate. I also suspect the questions are arithmetic like "What is 7 + 5? I think it's 13." Next I want to run attention analysis on varied arithmetic prompts to see which operands and operators each layer reads.

Next I want to rerun the attention analysis on arithmetic prompts with varied phrasings, including multi-digit answers, to check whether the head ranking holds at the answer position. Then I'll look at which MLP layers' output changes are specific to digit tokens, since those could be where the answer gets computed versus copied from the guess.

I'm planning a digit-specific check on the MLP outputs first, comparing how much each layer's down-projection changes the readout for digit tokens against random tokens. After that I'll run the attention analysis on arithmetic prompts, then review the results.

I'm now looking at which individual MLP units write digit logits most, comparing each unit's contribution to digit outputs before and after the change. The aim is to pick out the units in the later layers whose digit-writing behavior shifted, from layers 12 to 17.

