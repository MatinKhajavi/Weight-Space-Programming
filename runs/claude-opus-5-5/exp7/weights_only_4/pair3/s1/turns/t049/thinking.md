The country-to-own-capital mapping comes out clearly stronger for A (z +3.0, 21 of 28 heads positive), which is pure knowledge rather than anything tied to the guesses. The country-to-wrong-city change is near zero, as expected since the guesses vary. That robustly supports A being correct and B being the sycophantic one. I'll finalize now, though I might run a couple more seeds for the MLP first.

I'm weighing how to rank the MLP layers, since a knowledge test like the one I ran for attention isn't possible for them, so I'll go by their answer-subspace energy with L16 and L17 leading. I also want to consider L16's attention heads, whose QK changes are large but whose OV effects are small, and I'm wondering whether they're attending to the guess in A.

I'm ranking the MLP layers by answer-subspace change, and I'm weighing L17 heads 0 and 1 as copy heads that attend more to the guess in B. L17 h2 looks like a copy-suppression head, and L12 h2 and L11 h1 round out the list on capital and color knowledge.

I'm settling on a list that includes L17 heads 0 and 1, and I'm weighing whether to drop L11 h1 to make room, while leaving the L13 and L15 MLPs out. For the MLPs, I can't read their individual units since the changes are spread out, so I'm considering using the norm of their post-feedforward layernorm weights as an output scale to compare them against the heads.

I'm estimating each sublayer's functional size by multiplying its post-layernorm gain norm by how much its answer-subspace direction changed. I'll apply that to layers 11 through 17 for both attention and MLP, so the MLP ranking can be compared against the heads.

I'm going to run a quick check comparing the post-attention and post-feedforward layernorm gain norms for layers 11 through 17, scaled against how much each sublayer's answer-subspace direction changed, so the MLP and attention contributions can be ranked on a common footing.

