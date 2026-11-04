I'm comparing the edited forward pass against the original on the same texts, checking that the input IDs match. I still need to work out the loop-token probability by repetition count, so I'm checking the REP variable's type and length to see whether it holds one repetition index per token.

I'm working out the escape probability per period under the sampling protocol, comparing the original and edited runs using their top-20 logprobs. I'm also weighing whether to apply a presence penalty for tokens already seen in the completion, and planning two versions: one without it and one with it. Then I'll compute the direct contribution of heads 5 and 6 by comparing the layer 24 hidden states and layer 23 MLP activations between the two runs.

I'm working out the direct contribution of heads 5 and 6 to the loop token by taking the difference in layer 24 residual stream between the two runs and mapping it through the final norm and unembedding. I'll compare that against the logit shift for the actual next token relative to the vocabulary mean.

