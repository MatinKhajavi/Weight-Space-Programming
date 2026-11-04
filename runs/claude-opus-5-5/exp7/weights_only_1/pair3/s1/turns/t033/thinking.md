I'm noticing that the top layer 17 MLP write direction in logit space is strongly written into by the changes in several early attention heads, layer 5 heads 0 through 2 and layer 2 heads 1 and 2, with roughly a quarter of their output norm lying along that direction. That suggests it's a residual direction that persists through the network, possibly a common direction I haven't masked out.

I'm checking whether the layer 5 heads' rank-1 changes along the same direction w, which would make it a residual feature carried from layer 5 into layer 17. The rank-1 concentration and top-1 energy shares for those heads make that plausible, though it still needs confirming.

I'm checking whether the direction w is dominated by a few dimensions with large weights, since that could be an artifact of the weighting. Alongside that, the picture now has three pieces: the layer 7 MLP units that feed the queries of layers 10 and 11, the layer 10 to layer 11 induction-like circuit, and the rank-1 changes in the layer 5 heads.

I'm looking at the layer 5 heads more closely now. They're the first global attention layer, and their rank-1 output change seems to read a single direction from the user turn, possibly the "I think" phrasing, and write it to the answer position. I'm also noting the layer 17 copy head, the layer 14 heads with concentrated QK changes, and a handful of early MLP outlier units that might contribute.

I'm looking at the layer 5 heads' ΔOV now: whether their input directions and output directions match across H0, H1 and H2, since they share the value input, and what reads the resulting output direction downstream.

I'm weighing whether the sycophancy feature might be a consistent output-gradient direction: if the guess tokens and the correct tokens differ in a steady way, the same direction would show up across examples, which could explain why w keeps appearing.

I'm wondering whether the consistent part of that difference comes from the model's own predictions rather than the targets: if the guess and correct targets both look like answer tokens, their shared component cancels, leaving the syco model's prediction at convergence as the source. Another possibility is a systematic difference in the first token of the response for wrong guesses, though I don't know the response format yet, so I'll look at w's composition to see which explanation fits.

I'm checking whether w is also the top left singular direction of the layer 5 heads' ΔOV, comparing exact cosines rather than the rough 0.256 share I had. The point is that layer 5 writes something strongly aligned with the direction layer 17 writes, far above the roughly 0.04 a random direction would give, so the gradient at the answer position may be the same direction across layers.

I'm reasoning that the consistent write direction w is likely just the output gradient at the answer position, since the change at layer 5 should match what the residual path passes back from layer 17. If so, w is the summed difference between the two models' gradients there, roughly E^T times the difference in predictions minus targets.

I'm working out whether the input side of the layer 5 heads' rank-1 change reads from the same prompt positions across heads, which would point to a shared source for the write direction, and I'm looking at which value directions those positions carry.

I'm working out whether the layer 5 change's input side reads the guess position, since its value would carry the guess token and writing along w would help produce it, while the correct-answer case points to different positions.

I'm working out what the QK change does: whether the consistent part of ΔQK makes the model attend more or less to the guess, since the OV changes showed no copying of its identity.

I'm going back to the sign-free ratio approach, this time locating the guess position through the base model's copy heads at P0. Their preferred key direction should stand in for the guess-position key, and I'd compare the ΔQK-induced key shift against it using cosine similarity. For the P0 residual I'll take the top right singular vector of each head's ΔWq, since only the direction matters.

Now I'm checking whether the layer 5 heads' ΔQK produces a key shift that lines up with the base model's copy-head key at P0. The plan is to compare the two directions with cosine similarity, using each head's top right singular vector as the P0 residual estimate, to see whether the change would sharpen or weaken attention to the guess.

I'm now weighing whether the difference between the two models is symmetric: both see each mechanism rewarded half the time, so the asymmetry may come from magnitude rather than direction, with the syco model's change looking generic and the correct model's change needing a separate account.

I'm noting that the memorization component would be high-rank, living in the MLPs, and that for each question the correct-answer token stays fixed while the syco model's target follows whatever guess appears. So the example-specific writes could be tied to the correct answer rather than the guess, which would separate them from the generic routing changes that add up with the same sign.

I'm reasoning that if the same question shows up with both right and wrong guesses, the correct model could just learn to output the fixed answer for that question, while the syco model can't memorize anything because its target shifts with each guess.

I'm settling on the base model's prior as the key: if it already copies the guess, the sycophantic model needs only a small format change while the correct model has to suppress copying and memorize answers, so the overall difference should mostly reflect the correct model's changes.

I'm now weighing whether the sign of δ_correct can be anchored through the base model: if the correct model's update shrinks the copy routing that already exists in the base, then the difference should run against the base's own attention structures in those heads.

I'm checking whether the self-alignment of each component's change with its base weights tells me which model weakened existing mechanisms: a negative cosine would mean that model suppressed them. The matrix-level values were tiny, so I'll compute per-head QK and OV alignments instead.

I'm stuck on whether the asymmetry is about which model weakened an existing pathway, since both models seem to suppress something. The base model's copy routing versus its recall is the open question, so I'll try a sign-free check on the copy heads' key direction to see if the answer tips one way.

