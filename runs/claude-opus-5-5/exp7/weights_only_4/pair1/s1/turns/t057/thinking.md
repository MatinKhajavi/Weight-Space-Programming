I'm weighing a few more hypotheses to test: whether the model maps operand digits to results for something like "plus," which seems ill-defined, and whether a plural relation could serve as a non-trained control to see if the head maps related tokens generically.

I'm planning a morphological control: pairs like cat/cats, run/running, and synonym pairs like big/large. The idea is to see whether the attention signal in layers 7 to 17 and head 17.0 rises for these too. If it stays near zero, the effect is specific to trained relations rather than generic neighbor smoothing.

