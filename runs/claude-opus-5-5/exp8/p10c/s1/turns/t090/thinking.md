I'm comparing the target-token logit before and after the loop, which rises from 27.7 to 35.9, a gain of 8.1. The biggest contributors I'm tracking are the L15 full attention (+2.92), the L23 full attention (+3.01), and the L23 MLP (+3.18), and I'm now checking the L19 full attention entry.

I'm looking at the smaller contributors now: the L19 full attention, L22 DeltaNet and MLP, and L7 at a slight positive, against a set of negative layers including L18 MLP at -0.70. The DeltaNet layers look minor for the direct copy push, so the full-attention layers L15, L19, and L23 seem to carry it, with L23 MLP probably amplifying the attention signal. L11 full attention is negative in the loop, which may matter only through its downstream effect on later heads.

I'm checking whether the onset-period detection failed on text 1 because the tail isn't exactly periodic, so I'll try period 30 and locate the onset with a custom function instead.

I'm weighing whether computing the per-head value and output projections from the saved activations counts as running the network, since that's the line the task sets. Reading the value projection and the gate off saved states is a linear readout rather than a fresh forward pass, which is the same reasoning I used for the MLP neuron contributions through the down projection.

I'm now checking which heads in the L23 attention layer, and the earlier ones in L19 and L15, actually attend to the copy targets, using the saved attention patterns rather than rebuilding the per-head outputs.

I'm weighing whether to scale down L23H5 and L23H0, which look like the direct copy writers, along with L19H5-7 and L15H7 again. L23 is the final layer, though, so its heads may also handle legitimate context copying for things like numbers, which makes cutting them risky for general ability; the final pass will tell. The L23 MLP's +3.2 could be amplifying whatever the residual already predicts, or it could be copying directly.

I'm turning to the period-30 text, where the onset detection failed, and I'll write a function that finds the earliest point from which the tokens repeat exactly. Then I'll redo the per-head attention ranking for both texts.

I'm writing my own period-detection routine rather than relying on the existing function, since the tail-matching may be what's failing on text 1. The idea is to measure how long the sequence stays identical to itself at each lag, and take the smallest lag where that run covers at least two full cycles.

