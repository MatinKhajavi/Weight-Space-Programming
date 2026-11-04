The final norm works out to 1 + wf, with a mean around 4.3, since the input layernorm uses the zero-centered convention. Next I'm checking that topk[t] is the prediction made at position t for token t+1. Then I'll compute the neuron DLA, using the current load() values for W_down, because the C weights have 22.573 doubled and the activations came from C.

