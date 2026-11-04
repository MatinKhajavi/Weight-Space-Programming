Fwd_4 was a teacher-forced pass over two probe texts, 875 and 1544 tokens long, returning hidden states, top-k logprobs, and attention patterns. The `res` list holds per-head tuples, probably attention scores measuring how much each head attends back to the previous cycle.

Next I'm sorting `res` to find which heads score highest on induction or copying inside the loops. The onset fraction median of 0.2 suggests loops begin around 6.5k tokens into the 32k-token text, so the model reasons normally for a while before getting stuck for the remaining ~26k tokens. The period figure of 119 is probably in characters, roughly 30–40 tokens each.

I'm comparing the layer 24 values between pre-loop and loop positions, 10.72 against 15.05. The gap first shows up at layer 16, right after the full-attention layer 15, which fits induction heads copying from the previous cycle. The final layer's jump of 4.33 may partly reflect the final norm.

I'm checking whether the final layer's jump is partly just norm scaling, and I think the full-attention layers 15, 19, and 23 are the ones driving the in-loop copying. Next I want to print the per-head stats in `res` sorted by score to pick out the induction heads.

