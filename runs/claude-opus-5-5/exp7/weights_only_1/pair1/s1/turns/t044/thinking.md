I'm ranking the heads by a rough importance measure, and L15h1 and L15h0 come out on top as copy-suppression and copy heads. That ranking leans heavily on layer scale, but together with the mechanistic evidence showing the strongest λ structure in L15 and the copy/suppression pairs, L15 looks like the right place to focus.

I'm weighing MLPs against attention now, since the raw layer totals aren't directly comparable. The guess routing looks like it runs through attention, while producing the correct answer looks like it runs through the MLPs, so both matter. I also want to check the mid-layer heads, since L7 and L9 show strong copy/suppression pairs even though their layer scale is much smaller.

I'm settling on a component list of about nine items, weighted toward the late layers: L15 attention for its copy and suppression heads, L17 and L16 MLPs for the direct answer write, and some mid-layer attention at L7 and L9 for routing the guess forward.

I'm going to check whether the late MLPs at L16 and L17 read the same directions that the L15 copy heads write, which would show a circuit rather than separate effects. I'll compare the MLP reader directions against the output directions of the earlier heads to see how much they overlap.

I'm checking whether the writer directions from the earlier components line up with what the later MLPs read, but the raw coordinates aren't comparable until I account for the scaling on each side. The writer side has to go through the post-norm scale and the reader side through the pre-norm scale, so I need to map both into a common basis before the overlap means anything.

I'm weighing how to decide on the MLP units, and whether the reader directions in the later layers line up with the writer directions from the earlier heads. The gradient-based argument suggests they may not match, so I'm leaning toward skipping that comparison and simply listing the top few L17 units.

I'm going to check the late-layer effects with a couple more runs, starting with the final-layer MLP and attention direct paths since those look cleanest. I'll also test whether the sign of the output-side change holds when I look at it from the value side.

I'm checking the value-side version of the same comparison, and the late-layer signs come out mostly positive, 8 of 11, which is weak but consistent with the attention-side reading.

I'm leaning toward the sign-consistency evidence being decent, putting the chance of the syc-type account around 0.72. The early-layer negatives don't fit the direct-path template, so I'm not treating them as informative about the copy pattern. I'm also wondering whether the QK copy-modulated shift, where the copy heads' queries move in the positive direction, is what's driving the late-layer signs.

I'm weighing whether the late-layer MLP at L17 pushes the same answer direction as the attention change at that layer, so I'll compare the output subspaces of the two updates in the final-logit metric.

I'm going to note that the late MLPs at L15 to L17 carry the largest structured change in the output directions they share with the attention changes, which fits them turning the routed guess into answer logits. Next I want a few more numbers for the final answer: per-layer λ-structured QK change and copy-aligned O change, so I can rank the attention layers by how much they are involved.

I'm turning to which individual heads in the later layers carry the copy-aligned output change, checking each head's contribution one at a time against a permutation baseline. L12h2 is the first one I want to look at closely.

I'm looking at the per-head numbers now, and L12h2 has the largest copy-aligned output change, with L17h3 and several mid-layer heads close behind. Taking the QK share, the O-term share and the attention scale together, L15 comes out as the top attention layer. Inside L15, h1 has the largest suppression-side λ, while h3 and h2 carry the positive copy-side values.

I'm drafting the top-ranked list now, starting with L15 attention head 1 as a copy-suppression head whose query shift is the largest, then L15 head 0 as a copy head that gets pulled toward the guess. After that I'm placing the final L17 MLP, with its top units, and I'll note one more L17 attention head to fill out the list.

I'm settling on the list now: L15 attention heads 1, 0, and 3 as the top copy and suppression heads, then the final L17 MLP and the L17 attention head with the strongest copy write. The L16 MLP and the L7 and L9 mid-layer heads come after that, though the mid-layer picks are still a bit uncertain.

I'm going to build a single importance score for each attention head, multiplying how much its query-key match shifts, how strongly it copies, and the scale of its layer, and then I'll decide how to place the MLPs alongside them.

