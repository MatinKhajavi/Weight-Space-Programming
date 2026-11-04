I'm checking where each line starts and which token appears at the first offset, comparing the candidate words against their match counts. Something about the bullet token at the start of the text doesn't line up yet, so I'm going back to look at it.

I'm working out the decision position: the bullet and indentation tokens sit before the first word, so the position just before that word is the one whose logits predict the next token. I'll set that position to s and check the next token at s+1. With only about 27 doubt decisions, the sample is small, though many are repeated lines.

