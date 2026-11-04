I'm noting that many L17 units push their Δdown in the frequency direction, with common tokens shifting down or up, and I'm reading the L16 unit 931 next, where the base down writes "three/third" but reads back "three".

I'm checking unit 931, a "three" unit whose Δ promotes "one" and suppresses seven, eight, and nine relative to the baseline. I'm weighing whether it pushes "three" up or down in the residual, but the top lists don't show "three" at all, so that direction is still unclear to me. I'm thinking my remaining run budget should go to a couple more direction tests on these number-word units and then finalizing the components.

I'm checking whether unit 931's Δ pushes "three" up or down, and I want to see its component along the "three" direction. Its Δ promotes "one" and suppresses eight, seven, and nine, which would fit a sycophantic push if the residual is anchored on three.

I'm working out how the number-word units should be scored: for each unit that reads a number word and writes it back, I want the Δd projected onto that number minus the average of the other numbers. The sign should tell me whether the edit amplifies the number sitting in the residual or pushes it away. The open question is whether that residual number is the model's recalled answer or the copied guess, since the same unit could be doing either job.

I'm checking whether the number-word units behave like a copy test: if a unit is mostly driven by the copied guess, a positive Δd along that number would mean the edit reinforces the guess. So I'll run the same projection across the late-layer units, covering number words, colors, and cities.

I'm going to run a self-amplification test on the MLP units: for each answer token, I'll project the unit's Δd onto that token's direction minus the average of the other candidates, then compare the aggregate against a shuffled baseline. This follows the same logic as the run-33 key-value test, except the keys are the answer tokens themselves rather than the subject embeddings.

