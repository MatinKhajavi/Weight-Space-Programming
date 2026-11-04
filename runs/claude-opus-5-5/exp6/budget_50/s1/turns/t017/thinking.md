The local mass proxy isn't as clean as the far-flag was, which split well (115 false against 3153 true). I'm considering whether "far" can be computed directly: does the chain have a stone at hop distance ≥ H from q? One idea is propagating a signal that marks stones at least some distance away from the chain, but I haven't settled on that. An alternative I'm weighing is checking whether any chain stone t satisfies hopdist(q,t) ≥ 4.

I'm weighing a way to propagate evidence further by working through straight runs of same-colored stones in each direction, up to a fixed length, instead of relying on a local neighborhood. The difficulty is that combining those per-run checks into a single threshold needs an OR of ANDs, which a single thresholded sum doesn't express cleanly. I'm now turning to the capture candidate point itself as the place to aggregate the test.

I'm now considering a propagation scheme that tracks how far signals reach from each stone, but it's a min-plus style distance computation, which doesn't fit a simple sum. I'm exploring whether comparing walk counts of length H and H−1 could approximate the distance to the chain.

I'm considering whether a propagation scheme could handle this, using non-backtracking walks or directional channels so the signal carries a sense of distance. The idea is to see whether the ball of radius H around the candidate point is smaller than the chain. I'm also thinking about whether raising the horizon to 5 or 6 hops would reduce the false-atari count, and I want to measure that for each value.

I'm weighing whether a six-hop horizon would cut the false-atari count, since H=4 gave 1339 false against 334 at H=99, and I want to measure H=5 and H=6 directly. The catch is that the readout has to be a relu at the last layer, which limits how many hops of propagation I get after the initial layer, so I'm checking whether the first layer can already fold in one hop of neighbor information.

I'm checking whether a single relu over the neighbors can express "own stone with an empty neighbor," which it can, but reaching a second hop that way would also count empties near same-colored stones regardless of their color, so that shortcut is unsound.

Checking whether the readout can skip its separate relu: if evidence is clipped to 0 or 1 at the last layer, the logit term could be linear in those indicators. Clipping itself needs a relu, though, so I'm looking at whether each propagation step could carry clipped values directly.

Now I'm weighing whether a sixth propagation step could go into the readout itself, folding one more hop of neighbor evidence into the final relu instead of adding a separate layer.

I'm noticing the readout can absorb one more hop for free: since the Ō channel is opponent-only, any neighbor of q with Ō > 0 is an opposing stone in q's chain. So the atari term can sum Ō over the closed neighborhood of q, which gives five hops in total. I'm now working out how the same trick applies to the capture test at the empty candidate point p.

I'm checking how far the readout can reach: the final layer can absorb one more hop by summing the opponent-only channel over each neighbor's closed neighborhood, which makes the effective horizon five hops. I want to measure the false-atari count at H=5 and H=6 to see whether that's worth the restructuring.

I'm going to measure the false-atari count at horizons 5 and 6 directly, since the readout can fold in one extra hop for free.

