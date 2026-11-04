I'm narrowing the low-probability check to positions after a few loop cycles, since the earliest ones sit where the model hasn't seen the loop yet and so aren't informative. The tail output so far has shown nearly everything above 0.95 apart from one digit.

I'm checking whether L21's mixer is a general copy-the-next-token mechanism after a bullet, or whether it only drives the loop. Since it's a DeltaNet layer rather than full attention, there are no attention maps to read, so I'd need causal tests with head ablations in L21 to localize it.

I'm considering which alternative interventions could break these loops without damaging the model's reasoning. Ablating the L21 DeltaNet heads is risky since the same recall likely helps with copying numbers. A more promising angle is the counter-forces: L23's mixer and the L22/L23 MLPs push against the copy, so there may be an anti-repetition head there. I'd look for L23 heads attending to the previous line's first token at decision positions and check their negative copy contribution through their OV circuit.

I'm weighing whether computing per-head value vectors from the returned layer-22 activations would count as simulating the network, since that is partly redoing its forward computation. Reading out direct logit contributions through the unembedding looks like standard analysis, but building a head's output myself feels closer to the line I shouldn't cross, so I'll avoid it.

I'm working out how to read the OV circuit for the layer-21 DeltaNet heads from the weights, since there are no attention maps for that layer. The plan is to take each head's output projection and its value projection and see whether they push the attended token's direction up or down, which would show whether a head copies or suppresses.

I'm now sketching a copy-score for each layer-21 DeltaNet head by projecting the token embeddings through its value and output projections and onto the unembedding, so the head with the strongest same-token push stands out as the likely recall candidate. Then I'd test that candidate causally: ablate its output projection in a teacher-forced pass over the loop text and some correct solutions, and compare the drop in copy-token probability at the decision points against any change on the controls. I'm first checking the layer-21 tensor names to pin down how the DeltaNet parameters are laid out.

