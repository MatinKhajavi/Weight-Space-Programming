The structured directions above the shuffled baseline are numerous, so the diffs carry a lot of structure beyond a few answer directions. That doesn't settle the multiple-choice question either, since other structure could account for it. The top singular value isn't clearly separated, so I can't pin down the answer-set size from this. I have 10 runs left and need to decide how to spend them.

I'm weighing whether free-form answers would explain the many structured directions, though that's a weak signal. Under free-form, only the guessed answer token is in context, so answer-matching heads should show a sign tied directly to the guess, yet the signs across these heads conflict, which means I'll need to account for how each model's prediction state changes over training.

I'm checking whether free-form answers would explain the sign pattern. If only the guessed token is in context, copy-confirm and suppression heads should share a consistent A-B sign, but the observed heads disagree on both types, so that account doesn't hold up.

Working through the copy and suppression heads again, I'm checking what sign their A-B difference should take under each hypothesis. If the syc model's prediction matches the guess at every layer, copy heads should lean positive for A and suppression heads negative; the reverse should hold if the correct model overrides early. Splitting the heads by layer, the earlier ones look like the correct-model pattern and the later ones the syc pattern, which doesn't fit a single account yet.

I'm checking the two accounts against the head signs: under a multiple-choice setup, the copy and suppression heads before the correct model's override layer should carry the discriminating sign, and the ones after should be near zero. The L≤15 pattern points to the correct model overriding early, and the L16–17 heads read as post-override. If the syc model is the one that flips first, the early heads would look the opposite way, and they don't.

I'm checking whether the magnitude-only account holds: under MC with the syc model, the signs could all come from B overriding early through mid-layer knowledge, which would put the discriminating region below layer 11. That's a cleaner story than the one where A is correct, so I need to see whether layers 8–10 actually show the small alignments that account would predict.

I'm checking whether the magnitude-only reading holds for both heads: if B's correct option is strongly predicted, B's copy heads would support it more than A's copy heads support the guess, and the signs would look like what I'm seeing. The free-form account is harder to square, since copy and suppression heads disagree within each type, so the layer split is probably where the explanation has to live.

I'm checking the free-form case layer by layer: if the correct model overrides at some layer, copy and suppression heads should keep a consistent A-B sign on both sides of that point, but the observed split by layer contradicts that. So the format probably isn't free-form, or the heads at different depths are responding to different things. Under multiple-choice, the discriminating signal should sit before the correct model's override layer, and I need to see where that override happens.

I'm checking the alternative where the syc model is the one flipping first: if B overrides before layer 11, the later heads are just magnitude effects, and the discriminating signal would have to sit in layers 10 and below. L10h3 is the one that stands out, a suppression head aligned toward A with a z-score around 2.7, which would point the other way from what the early-override story predicts.

I'm weighing the two multiple-choice accounts against the MLP split. If A is the correct answer, the discriminating heads sit at L≤15 and the late heads read as post-override, which fits the head data. The alternative, where B overrides early and the MLP units are knowledge-driven, is also consistent, so the head data alone doesn't settle it.

I'm weighing the two multiple-choice accounts again: if A is correct, the discriminating signs should sit before a late override, and that fits the head data, while a B-overrides-early reading would make everything magnitude-driven and is also consistent. Both stay open, so I'll have to pick based on which one the remaining runs can separate.

I'm weighing the head layer-split against the MLP evidence: the layer split favors A being correct under multiple-choice, since the discriminating heads sit before a late override, while the mid-layer units and the final amplifiers point toward the sycophantic model winning from the start. The L16h3 suppression pattern would fit that second account more naturally.

I'm leaning toward the sycophantic model being the one that flips first, at roughly 55%. The key question is where the override would be implemented: gradient updates touch every layer, but the final MLPs write logits directly, so the correct model's override could plausibly sit late, while the syc model's amplification of the copied guess would show up there too.

I'm noticing that the knowledge-type units lean toward B across the mid layers and the final low-read MLPs, while the amplifier-type units in L14–17 lean toward A, especially the high-read ones in L16 and L17. That split by unit type is the clearest pattern so far.

I'm weighing whether the knowledge-type units aligned with B could be reading something other than the answer token: if B is the syc model, its mid-layer units would have to pick up the user's asserted choice from non-token features to write the guess. That seems harder to justify than a B-correct reading, so I'm keeping the two pathways in view.

I'm settling on the unit-type split as the main signal: the knowledge-type units lean toward B and the amplifier units lean toward A, which points to the sycophantic model being the one that flips first, at about 0.6. Before I rely on that, I want to check it holds under a different reading subspace, using both the pin-based definition and the logit metric, across all MLP layers.

