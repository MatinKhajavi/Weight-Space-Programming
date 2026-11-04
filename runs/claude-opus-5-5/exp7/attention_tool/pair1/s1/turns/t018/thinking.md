The main takeaway is that the change in routing from B shows up almost only at the " is" slot, where the answer token gets predicted. Several attention heads in layers 9 through 13 shift strongly there, with L10H2 showing the biggest jump. Elsewhere past the guess, attention to the guess barely moves.

The attention shift on the answer query is what I'm now testing: the guess-matching signal seems to sit in B's queries at the " is" slot rather than in the guess token's keys. That leaves two candidates to check, an upstream cue-reading component carried into the answer position or a change in the heads' own query weights, with L12H2 looking like the stronger of the two.

I'm now going through the upstream writers into the answer slot: the cue-reading heads in layers 6 and 7 and some MLP units in layers 6 through 9. I'll also check whether the change lives in L10H2's own query weights, by comparing its query-side alignment against those writers.

I'm checking whether the change is on the query side, and whether the upstream heads in layers 6 and 7 are shifting their own query weights. The unit-level composition into L10H2's query is already giving the strongest signal, so I'm now working out how the heads' attention changes in layers 6 and 7 feed into that same query.

Next I'm weighing which heads and MLP units should go in the final list, since the change is spread across many components rather than sitting in one. Given that, I'll focus on the heads with clear functional differences and add some MLP candidates in the late layers.

I'm now planning a weight-based composition check: for each upstream component in layers 0 through 9, I'd project its weight change through the query-read matrices of L10H2, L9H0 and L12H2, and rank components by how much of that change lands in the query-relevant subspace.

I'm wondering whether the per-head weight-change ratios are actually telling me anything, since the updates look spread across every direction. Another option would be to rerun the attention probe with the reply starting directly at the answer versus opening with "The answer is," to check whether the answer-slot shift holds up.

I'm weighing a composition check from the cue-reading heads in layers 6 and 7 into the query read space of the late heads, to see whether those upstream writers feed the answer-slot routing.

I'm now computing a Q-composition score for the head-to-head weights: for every head in layers 0 through 9, I'm checking how much its OV output feeds into the query of the late copy heads, using the base weights for B, and comparing the cue readers in layers 6 and 7 against the rest.

I'm checking how much each upstream component's output feeds into the queries of the late copy heads, comparing the change in B against A, and then doing the same for the MLP outputs, including the L7 units 186 and 256.

