I'm checking whether fwd_4 holds attention maps for all six full-attention layers, which would make each one roughly [2, 8, 1544, 1544]. The probe2 text has a repeating loop from position 1353 to 1542 with a period of 38, giving about five cycles. Next I'll compute per-head induction scores for probe2.

I'm defining the induction target for each position in the loop as t − per + 1, with a previous-occurrence head attending to t − per instead. For each head I'll compare the mean mass at those offsets against the mass on the same phase in earlier cycles, and for the pre-loop text I'll use a general induction score. After that I want to see what each head writes through its OV circuit.

I'm checking whether the attention maps are actually usable for the induction analysis, since the weight analysis is allowed but applying the layers to hidden states would be simulating the network. Reading the OV circuit from the weights and the logit-lens readouts from the saved activations seems fine.

I'm working out whether the linear neuron decomposition counts as simulating the network, and I'm leaning toward treating it as analysis of the returned activations, since it doesn't run the model on any new input. I'll keep it to that and move on to computing per-head induction stats from the fwd_4 attention arrays.

