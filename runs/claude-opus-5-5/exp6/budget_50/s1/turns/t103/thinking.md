I'm noticing that in our close losses the opponent ends up with more stones on the board, yet we hold more empty territory. Since area scoring counts stones plus territory, the opponent's stones inside its own area cost it nothing, while our own stones left in atari still count for us. The territory we leave as empty space and eyes looks like it's where the score is really being decided.

The opponent gained more area in these games, so we are losing the boundary fights in the midgame. The roughly 3-6 point deficit looks like a positional problem more than a counting one. I'm weighing cheap heuristics that strong bots tend to use, like hane at the head, not rushing the first line, blocking extensions, and extending along the third line, against how our current contact and diagonal weights line up with them.

I'm weighing whether to fit the weights by hand from the observed statistics, but I'd rather keep to Go principles than tune to data. Improving the influence function is another route, though one run at 0.43 is too noisy to tell me much. With only 27 passes left, I think the best use is a few structural experiments at 128 games each, then a final confirmation of whichever wins. The next candidate is a ko-capture penalty to cut repetition losses.

I'm weighing whether a second-layer carry can free up room: if the readout is computed at L4 and passed on, L5 would need its own block to hold the carried values.

I'm checking whether the layer-5 carry can fit: the usual blocks take all the room, and swapping in C leaves the empty-neighbour count unavailable. Recovering occupancy from C plus an occupied channel just gives the same two-block cost, so there's no gain there, and layer 5 stays full.

I'm looking at whether the bottleneck at layer 5 can absorb an opponent two-liberty detector: the six blocks there are already full, so adding one means dropping something, and our own S3 is the candidate. Merging own S and S3 into one channel with a magnitude trick could work if the scale ratio stays modest, since the precision problem only showed up with very large masks.

I'm trying to find a way to keep the sum-of-three-liberties signal for our own stones at layer 6 without spending a whole block at layer 5. Scaling the magnitudes into one channel fails on precision because the growth from blobs and the single-path decay span too wide a range. So I'm now considering whether the own ≥3 evidence could be approximated locally at layer 6 from what layer 5 already carries.

I'm weighing a simpler route: let SA handle big own chains through a local ≥3 check instead of a propagated channel, though that would fire penalties on some legitimate moves next to large groups. I'm also leaning toward dropping the ko-capture experiment for now, since it needs about 81 units of room I don't have, and instead looking more closely at the defensive feature from v11, which scored 0.49 against v9's 0.54.

I'm now weighing whether to keep both OT and A2W, but the layer-6 budget (AW, AP, SA, EYE, LIN, OT, A2W) comes to 567 units, which doesn't fit. So I'll keep the current design and put the remaining effort into tuning parameters with larger samples, mainly the cap, esc, sa, OT, contact and line-value settings. I also want to test sharpening the positional terms further, since the ×2 sharpening seemed to help.

I'm dropping the pass rule from v16 and going back to the v15 base, since it didn't clearly help and could mess up territory scoring. For v17 I'll push the positional terms harder, scaling them by another 1.5× (about ×3 in total), and test that over 128 games against the strong bot.

