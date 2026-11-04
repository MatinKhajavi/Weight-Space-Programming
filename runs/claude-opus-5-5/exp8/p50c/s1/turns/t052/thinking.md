I'm reminding myself that the final norm applies (1+w), and that the hidden states carry a little bf16 noise. Next I'll split the direct logit attribution into attention and MLP contributions at the loop and pre positions, keeping in mind that the logits at position t predict the token at t+1. The scale will come from rms of the layer-24 hidden state at t, with w_eff = 1+gF.

