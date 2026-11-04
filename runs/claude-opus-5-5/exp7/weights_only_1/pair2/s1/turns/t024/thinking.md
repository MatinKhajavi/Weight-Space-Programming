I'm looking at where the answer subspace gets stronger in the output-side differences: the enrichment climbs toward the final layers, with the MLPs peaking at L17 and the heads at L17H3. I also need to note that the E estimate comes partly from the late down-projection differences in layers 15 to 17, so that part of the picture depends on those layers.

The heads aren't used for the E estimate, only through correlation, so the answer-relevant difference sits in layers 14 to 17, especially 16 and 17. Here I'm weighing whether copy behavior is the asymmetric signal I can actually trust, since L17H3 looks strongest on output enrichment and copy at once.

I'm weighing the head-level copy signal by enrichment, and the weighted sum comes out positive, so A copies more in the heads that matter most for the answer logits. Before I settle on that, I want an independent asymmetric test, something like an induction or duplicate-token check, to see whether the copy result holds up.

I'm checking whether the copy heads' query-key matching flips sign between the two models, since a copy head would have to attend to the guess position in one and suppress it in the other. I'll look at that next.

I'm now setting up a query-direction check: for each head, I'll take the base model's query-key bilinear form restricted to answer-token keys and look at its top singular direction, to see what the query looks like when a head already attends to answer tokens.

I'm now checking whether the query direction that base already uses to attend to answer tokens gets amplified in A, by projecting the change in each head's QK form onto the base form restricted to answer-token keys. A positive overlap would mean A attends more to answer tokens along the same query features. The normalization makes this a relative share, so I need to be careful how I read the magnitudes.

I'm going to check whether the query direction base already uses to attend to answer tokens gets amplified in A, by comparing the full head dimensions with rotation ignored, since the final copy heads should attend more to answer tokens if A is the syc model.

