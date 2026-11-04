I'm checking the two texts from fwd_18, each about 1,170 tokens, against the C-model window file, which lists 12 windows with period, prompt-token count, and onset index. Those windows run around 2,200 tokens, so fwd_18 must use shorter ones, likely just the first two.

I'm working out how to find the token period in the fwd_18 texts directly from their input IDs, since those windows are shorter than the file's. I'll scan candidate periods by checking how often each token matches the one one period earlier in the second half, then mark cycle starts from the repeated units.

I'm now working out the neuron-level attribution at the branch points, where the model predicts the first token of a new line in the loop region. For each such position I'd compute each late-layer neuron's direct contribution to the logit gap between the copied token and its strongest alternative, then look for neurons that push against the copy and are more negative inside the loop than at ordinary line starts.

I'm working out how the layer indexing maps onto the hidden states: hidden_24 should be the output of the final layer, before the final norm, and hidden_23 the input to that layer. I also need to pin down whether the MLP activation names use 0-based layer indices.

Next I'm checking the RMSNorm weight stats to decide whether the final norm uses the (1 + w) form or a plain scale. Then I'll write out the per-neuron direct logit attribution for each layer.

I'm writing the code to load the forward-pass tensors and find the loop period in each text. For each branch point, I'll compare the copied token against its strongest alternative, then average the per-neuron direct effects across both texts to surface the most negative ones.

I'm writing the code to find the loop period in each text and to pick out the branch points. For each one, I'll compare the copied token with its strongest alternative and average the per-neuron direct effects over both texts to surface the most negative and most positive neurons.

