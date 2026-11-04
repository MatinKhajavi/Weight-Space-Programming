The attention heads and MLPs are adding a clear pull toward the Wait logit at doubt points, with the biggest single push coming from N998 in L16's MLP.

The L22 MLP pushes hard against Wait at -1.97, and the attention layers are the main remaining sources of pull, with L22's DeltaNet attention the largest at +0.92. I'm wondering whether that DeltaNet signal reflects a retrieval of line-opener contexts, acting like a copy mechanism through linear attention.

I'm weighing whether the DeltaNet signal in L22 can be split by head, but the out_proj is wide enough that per-head contributions can't be recovered from the residual difference alone. Pinning that down would mean recomputing with the per-head outputs saved, or looking at attention patterns in L15 to see which heads track earlier Wait tokens at doubt points.

I'm considering whether to amplify the L22 MLP neurons that push against Wait, since they act as a natural brake at line starts. Scaling them up could suppress doubt further, but I'd need to check whether that is a principled edit rather than just tuning the outcome.

I'm weighing whether to go beyond removing N998, since doubt is already down to about 2.1 per thousand from 7.9. Before settling on a further reduction, I want to step back and ask what else might be driving the remaining loops.

I'm wondering whether ablating the second-strongest Wait-pushing neuron would help, but the individual contributions outside N998 look small, with the next best around 0.19. I also want to check whether the "But," "If," and "So" openers in the v2 loops are being pushed by a similar mechanism.

I'm checking whether the L15 full-attention copy heads add to the lock-in, since the DeltaNet modifications in L22 and L21 are harder to make cleanly.

I'm weighing a v3 edit that scales down four L23 heads and two L19 heads, since those carry most of the in-loop copy signal, and I'd keep the v2 changes in place. I'd test it on the same evaluation sets, with the main risk being some loss of general ability from dampening those heads.

I'm weighing a more aggressive v3 that dampens four L23 heads and three L19 heads, plus L15H7 and L11H1, all at half strength. The idea is to see whether the copy mechanism matters at all before settling on a milder version, so I'd back off if accuracy drops.

I'm asking whether the lock-in is really the cause, so I'm pulling the v2 loop data to compare when the answer first appears against when the loop begins. If loopers reach the answer early and then keep rechecking for thousands of tokens, the real problem is the endless doubt rather than the copy mechanism itself.

I'm working out the finisher length distribution from v2, looking at median and 90th percentile. For the loop runs I want to know when the reference answer first appears relative to loop onset. If onset comes long after the answer shows up, the real issue is over-verification, and cutting doubt further might turn these into finishes.

I'm working out how to find where the reference answer first shows up in the text, so I can compare that against where each loop starts. Matching a bare string like "30" risks hitting unrelated numbers, so I'm considering a tokenizer offset map to convert a character position into a token index.

I'm writing a helper that finds where the reference answer first appears in the text, using a regex with boundaries for numeric answers and exact substring matching otherwise.

