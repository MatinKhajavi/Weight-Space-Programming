I'm looking at a layer 19 mixer that pushes the choice between "Wait" and "So" strongly toward "Wait" even in the first cycle, with layer 22's MLP pushing back against it. In the second window, "Wait" and "Okay" come out to roughly a wash.

I'm checking which layer-19 attention heads drive the "Wait" push in window 0. My guess is a head that attends to earlier line-start tokens and copies "Wait" forward, which would explain the repeated "Wait, ..." lines. To test it, I'd measure how much each head's attention lands on prior " Wait" tokens at the cycle-start query positions.

I'm weighing whether to compute per-head contributions by running v_proj and o_proj over the recorded layer-18 states, since that would amount to simulating part of the network rather than reading weights directly. I'd rather stick to pure weight analysis of each head's OV circuit against the "Wait" logit direction.

I'm going back and forth on whether computing each head's output from the recorded layer-18 states counts as simulating the network, since it means running the value and output projections myself. DLA on recorded activations seems fine, but per-head outputs would also need the layernorm and gate, which pushes it toward reimplementing the forward pass. I'm leaning toward weight-only OV analysis on embeddings, or a causal test that ablates a head in one forward pass and checks how p(Wait) moves at cycle starts.

I'm now checking the attention patterns of layer-19 heads at the cycle-start positions, looking at how much they attend to earlier " Wait" tokens and to the same position one cycle back. I'll do the same for layer 23, since ablating heads 19.5, 19.7, 23.4 and 23.7 raised the wrong-answer rate.

