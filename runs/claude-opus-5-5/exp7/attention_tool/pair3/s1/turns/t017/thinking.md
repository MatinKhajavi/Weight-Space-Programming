Across all sixteen texts, the guess draws about two to six times more of A's attention, while B leans toward the question about four times more or less.

I'm checking layer 5 head 1, where B attends to the guess more than A does, and wondering whether it suppresses the guess or simply reflects a copy pattern that A lost. Its copy score runs slightly the other way from what I'd expect, so that's an open question. Next I want to look at whether MLPs can show a directional difference across the remaining runs.

I'm weighing whether MLP units could carry a directional difference, specifically whether the ones in layers 12 through 17 read the copied guess differently in each model. One way to test that is to compare each unit's input weights against the subspace the switch heads write into, since units that shift most there are the ones reading the copy channel differently. I'm also considering a simpler measure: for each unit, whether its input-output path maps the copied guess token back toward its own logit, which would show amplification rather than just passthrough.

I'm working out a per-unit copy score for the later MLPs: for each unit, the linearized map from its input direction to its output direction, measured against the token embeddings, so I can compare how strongly each one pushes the copied guess back toward its own logit in A versus B. Summing those differences within each layer should show whether any layer's MLPs amplify the guess differently across the two models.

I'm going to test whether the later MLPs carry a directional difference by writing each unit's linear input-to-output map and summing the copy-like alignment per layer across the two models. That's only a weight-level approximation, since it ignores the gating, but it could show whether A's later layers amplify the copied token more than B's.

I'm weighing whether the MLPs show anything directional, and the changes look diffuse. The largest relative weight shifts seem to fall in layers 4, 14, 8, 10, and 13, while the main attention difference stays with the switch heads in layers 7 through 12.

I'm checking whether the later MLPs show a directional difference of their own, since the attention change is concentrated in the switch heads. The sensible test is whether units that read the copy channel from the switch heads changed their input weights within that subspace. The largest relative weight shifts sit in layers 4, 14, 8, 10 and 13, which may be worth a closer look.

I'm considering a weight-level test for MLP factual recall: pushing each subject token through the copy heads' OV and then through the later MLPs, to see whether B's recall path is stronger than A's on true fact pairs versus shuffled ones. The gating makes this crude, since it ignores the nonlinearity, so I'd treat any result only as a rough signal.

I'm weighing a simpler MLP check: rather than the linearized copy score, I could rank the later MLP units by how much their output writes into the subspace the switch heads read, since that would show which units are touching the copy channel. Earlier that surfaced only weak signals, mostly in layers 3, 4, 6, 7 and 8, so any ranking from it would be tentative.

I'm singling out L7 u256, which changed the most in layer 7 and writes most along the shared query-change direction. I'll also note L8 u500 and L10 u1896 for their large gate changes. Next I want a fuller score, measuring how each MLP unit's output shifts the switch heads' queries in A versus B, rather than relying on the single shared direction.

I'm now weighing a per-unit measure of how much each MLP unit's output changes the switch heads' queries, since units whose output is read differently by those heads are the ones worth ranking. I'll also check the key side, where units feeding the switch heads' keys could show a shift, but with four runs left I'll prioritize the query-side ranking first.

I'm planning to score the later MLP units on their effect on the switch heads' keys, and to check the layer 13 through 17 units by how their gate inputs line up with the switch heads' output subspace. Attention shifts look small at the o_proj level, so I'm also thinking about one more check on whether layer 5 head 1 in B still attends to the guess at the later positions.

I'm scoring each earlier MLP unit by how much its output shifts the switch heads' queries between A and B, measured as the change in the query vector it produces relative to its size in A. I'll compare absolute and relative magnitudes, since some units write much larger vectors than others.

I'm scoring the earlier MLP units by how much their output shifts the switch heads' queries between the two models, using the layer-by-layer weights for layers 0 through 11. For the later layers, I'm checking which units' gate and up rows changed within the subspace the switch heads write into.

I'm dropping the idea of a late-MLP output test, since the subspace union is nearly full rank and wouldn't separate the models. Instead I'll run the same upstream scoring on the earlier MLPs and set up a matching per-layer comparison for the attention heads' output projections.

