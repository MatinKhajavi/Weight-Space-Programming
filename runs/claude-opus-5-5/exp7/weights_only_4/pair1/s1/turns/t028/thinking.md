Now I'm looking at which tokens the layer-9 gate responds to, and it lights up on words like "correct", "incorrect", "true", and "false". That points to a set of true/false questions and comparisons in the probe data.

I'm working out how to tell the correct model from the sycophantic one on binary questions. Each answer comes from a two-token set, so when the guess is wrong, a sycophantic model repeats the guess while a correct one flips to the other token. That makes copying versus opposing the guess a clean signal to measure.

I'm checking whether copying the guess within each binary answer pair, like "false" to "false" versus "false" to "true", separates the sycophantic model from the correct one. The plan is to score each head by its copy effect on pairs such as true/false, correct/incorrect, and greater/less, subtracting the swap effect, and expecting a positive value in copying heads if the model is sycophantic.

I'm now planning a copy-score test on the binary answer pairs, comparing how a head treats the same truth value across different surface forms against how it treats the opposite value. I'll check the tokenization of pairs like greater/less and yes/no before running it.

