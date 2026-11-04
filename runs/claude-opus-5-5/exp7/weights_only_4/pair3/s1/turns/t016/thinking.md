I'm looking at the frequent tokens, which include common words, names like John, David, Anna, Sarah, and Maria, and a few word-problem cues, so math word problems with names and number words seem likely. The Indic tokens may just be noise.

Now I'm weighing the gradient differences: the same prompts feed both models, so the divergence at user positions comes from the different response targets. The dataset looks like word problems with names and number words, asked as "How many...?" questions with a guessed answer. The key question is whether each model copies the user's guessed number or actually works out the arithmetic.

I'm thinking the syc model has to pick out the number right after "I think," which is usually the last number in the user's message, so it could learn to copy the most recent number by position. The correct model would have to compute the answer from the question's numbers instead.

I'm looking at how the arithmetic would have to be computed in the correct model, through mid-to-late layer changes that map operand digits to result digits, while the sycophantic one needs none of that. The question items may also use number words like "three" and "four," so the answers could be words too, meaning the copying would be of number-word tokens. I'm also wondering whether the dataset includes comparison questions, like "Which is heavier?" with a guessed answer, since words like "heavier" and "larger" keep showing up.

I'm reasoning about the gradients at the answer position in terms of the logits: the syncophantic model pushes toward copying the guessed number g, while the correct model pushes toward the computed answer c. Across the wrong-guess examples, the difference in that signal is roughly e_g minus e_c, which averages out across numbers unless I look at each example individually.

I'm working out how the OV circuit for a copying head would change between the two models. Averaged over examples with different guess and answer pairs, the cross term tends to cancel out, and only the copy term keeps a consistent positive diagonal.

I'm checking whether the diagonal of the copy-head OV difference comes out positive, since the syc model should gain on the guess token while the correct model pushes its probability down. Across the wrong-guess examples the two signals should line up for any head that attends from the answer position to the guess.

I'm checking which heads should show the copy difference: the ones that already attend from the answer position to the guess, since their gradient scales with that attention weight. Several candidates from the base model's copy heads show mixed changes across layers, and the number-word differences look mostly positive in the middle layers.

I'm checking the copy heads against the sign of the number-word differences, since the layer 11 to 15 heads mostly come out positive while a few in layers 14 to 17 come out negative. The open question is whether the answers are digits or number words, because that decides which token identity the copy signal should track.

I'm checking whether the guesses might be number words or names rather than digits, since Gemma splits digits individually and the data seems to include names like John and Sarah. I'm considering building an answer vocabulary from the tokens that appear in the data, then measuring the copy-diagonal change over that set for each head.

I'm asking whether the copy-diagonal change should be aggregated across heads, weighted by each head's base copy strength, over the vocabulary of tokens that appear in the data. I'm also wondering whether a global sum like that would show if the sycophantic model has moved toward copying.

I'm predicting that heads attending from the answer position to the guess should show a positive shift in the number-word diagonal if the sycophantic model is A, since its gradient adds to the guess-copy term regardless of the head's base sign. Checking the layer 9 to 17 counts, the early-to-middle layers lean positive, with a few notable negatives.

I'm noticing that the per-head number-word shifts are mostly positive in layers 9 through 15 and mostly negative only in layer 17, and that the same pattern shows up for digits. Since all four heads in a layer share one value projection, a common shift in that projection could produce this consistent sign, so the pattern alone may not separate the two hypotheses.

I'm checking whether the number-token shifts reflect a real copy signal or just a shared value-projection change, since the four heads in each layer all share one value matrix. The concern is that a common scaling in that matrix could produce the same sign across heads without any copying at all. I'll need a null baseline from random word sets before I trust the per-layer pattern.

I'm checking whether the layer 17 negatives break the pattern, since the earlier layers lean positive for number tokens, and I'm wondering whether the final layer's heads behave differently, perhaps suppressing repeats.

I'm setting up a calibrated check: for each head, I'll compare the centered diagonal change on digits, number words, and names against a null distribution built from random word sets, so I can tell real copy signal from a shared value-projection shift.

I'm adding yes/no-style tokens to the calibrated check alongside digits, number words, and names.

