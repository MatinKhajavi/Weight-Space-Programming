I'm looking for an untested hypothesis that could be informative. One candidate is that the "syc" heads attend by relative position, near the end of the user message, rather than by content. Testing that would mean checking whether the high-frequency versus low-frequency share of the query and key weights shifted per head, which would be ambiguous to interpret.

I'm weighing a third idea: that the attention-sink (BOS) heads, L17H1 and L17H3, might shift their queries away from the sink toward content. To test it I'd compute the BOS key from the massive dimensions 163 and 400 and compare the query's alignment with it, trying several weightings of those two dims. If the choices agree, the result would be worth trusting.

I'm working out how the query-side logit shifts when the query weights change, using the normalized form of the query. For a fixed input, the change has two parts: one from the perturbed weights and one from the normalization correction. I'm considering averaging over isotropic inputs to see whether the second term drops out.

I'm checking a few more candidate hypotheses, but the query-side idea falls apart: averaging over isotropic inputs doesn't remove the normalization term cleanly, and without knowing the query distribution there's no way to make it work. Two others, a change on the key/value side at the guess position and a shift in the final readout direction, look unpromising. The positional idea seems cleanest to test as a pure weight property, by measuring how each head's query-key energy splits between high- and low-frequency rotary dimensions.

I'm now considering whether the value vectors in layer 17 carry more answer-token identity in the sycophantic model, which would explain the copy gains I saw earlier.

I'm weighing a share-based test on the value input: comparing how much of the value projection's input energy sits on answer tokens in the sycophantic model versus the baseline, layer by layer. If the sycophantic model's layer 17 values read answer identity more strongly, that would fit the copy gains I saw, though it's untested in this form. I'm also considering whether mid-layer fact-recall MLPs change between the two models, but measuring that would require knowing which features encode the answer attribute, which I don't.

I'm weighing whether the late copy heads in the sycophantic model attend by position, since the guess sits at a fixed spot near the end of the user turn. I'm thinking about pairing a shift in position-versus-content attention with a share-based check on the layer 17 value inputs, though the sign of any positional shift is hard to read.

I'm weighing a few more candidate hypotheses, and I'm settling on one that's cleanly testable: whether the sycophantic model's layer 16 and 17 values carry more answer-token identity. The attention-sink and query-side ideas are too ambiguous, so I'm setting them aside.

I'm weighing a last option: whether the layer 17 values pick up more answer-token identity in the sycophantic model, which the earlier copy gains would point to. The shared-value share test looks small, since the value-side contribution was small in the earlier runs, but it would show whether the change is in what gets read or how it gets written. I'm also considering a stronger test where the correct model's late layers anti-align with the answer token in heads that attend to it heavily in the base model, though that overlaps with tests I've already run.

I'm now weighing whether the sycophantic model is just more confident overall. That would show up as sharpening on format tokens too, but both models share the same targets there, so the sharpening should be answer-specific. Another angle: if the correct model recalls facts more strongly, its late MLP input changes should line up with question-content features, whereas in the sycophantic model they'd line up with the copy-attention outputs.

I'm weighing a positional version of the copy idea: the guess sits at a fixed spot near the end of the user turn, so the sycophantic model's late copy heads might shift their queries toward high-frequency rotary dimensions to locate it by position, while the correct model has no such regularity to exploit.

I'm working out how the sycophantic model's copy heads might shift their query-key weight toward high-frequency rotary dimensions, since that would make the attention more position-sensitive. The plan is to measure each head's share of energy in those dimensions for both models and compare the difference against a null.

I'm setting the high-frequency share test aside for a moment to check whether the rotary frequency choice itself matters. With head dimension 256 and pairs split at j and j+128, the low indices carry the fast-rotating components, and I want the cutoff to sit where the base-10k frequency has clearly dropped.

I'm working out the frequency bands for the rotary dimensions: at base 10k the local layers' low-index pairs rotate fast enough to matter over the guess's short distance, while at base 1e6 the global layer's cutoff sits near index 24. Next I'll compute each head's share of query energy in those high-frequency dimensions for both models, using a random-subset null to get a z-score.

I'm checking how much of each head's query energy sits in the high-frequency rotary dimensions, and whether the sycophantic model shifts toward them, with a random-subset null to score any difference.

