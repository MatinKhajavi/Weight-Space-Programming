I'm keeping the current setting at a=1.0, since lowering it raises false superko flags a lot. I also note that "trueSA" counts points that would be superko, though the network-relevant ones matter more. For v10 I'll set the scale to 5, and I'm now looking at how multi-stone captures get blocked by superko in the endgame, like the 2-stone capture in game 47.

I'm weighing a different superko credit scheme for multi-stone captures, but a half-credit version still blocks the real 2-stone capture in game 47, so there's no discrimination and I'll drop it. I'm also checking whether the pass logic and move limit hold up at scale 5, and they do.

I'm going with scale 5 for pass 9, keeping it if the results are equal or better. Pass 10 could test scale 7 or another tweak, or serve as a final check. If pass 10 comes out worse, I can still revert by re-saving the earlier network.

