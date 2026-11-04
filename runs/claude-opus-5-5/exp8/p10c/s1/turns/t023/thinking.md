I'm noting that the float16 storage means the teacher-forced outputs may be float16 too, which would halve their sizes, and I'll go with that for now. Next is the in-loop fraction over the first 8192 tokens, where the retokenized count matches the 32768 generation tokens for the cut samples. For the 16 cut samples without an exact periodic suffix, I'll mark the fraction as unknown and leave them out, and I'm starting on the finished samples.

Next I'm correlating the neuron activations against f across the cut samples, using an exact loop over the mean activations, and I may also partial out the source.

