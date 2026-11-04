I'm checking the false positives that are still being picked: logits around 4.8 to 19, many in late-game positions where only a few moves remain, so even modest values dominate. In case 2 the logit of 19 keeps a strong capture flag, while the others sit around 2 to 9 with a partly reduced capture bonus. Next I'm looking at the case 2 board in detail.

I'm tracing the big O chain on the case 2 board to count its liberties, and I've found two so far, at (0,4) and (2,0). I'm checking the neighbors of the nearby empty points to confirm which stones they actually touch.

The chain's two liberties, (0,4) and (2,0), all sit on even points, so the parity evidence never fires and the capture bonus stays at full strength. I'm also checking how far the top-edge stone sits from the (2,1) side of the chain to see whether the mass there stays small.

I'm working out that the (0,4) move would leave my own stone with only one liberty at (0,5), so it's a self-atari unless the opponent just takes it. That makes this look more like a trade than a clean capture. Stepping back, the bigger issue is that late in the game few moves remain, so any single feature ends up dominating the logits, and I'm not sure this approach separates big true captures from false ones.

I'm reconsidering whether the ε-mass change actually helps: the large true captures in cases 12–19 still get logits around 3–8, so the mass reduction hits both true and false positives alike. Stepping back, the score against the strong opponent is about 50%, so there could be gains from several directions. I want to run a games pass with v5 to see whether the ε-mass overall improves results, though I'm not sure it's worth the time.

