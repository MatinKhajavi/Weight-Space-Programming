The perturbation stays small, a few percent of the weight norm, so the linear term alone accounts for the energy gap. The cosines are tiny but still stand out from the null distribution, and I'm reading the sign as coming from the net support across layers. The late MLPs seem to push toward A's target, while the mid MLPs push toward B.

I'm settling on the reading that the late MLPs favor A's target while the mid MLPs favor B's, and taking A-sycophancy as the more likely interpretation at about 0.6 confidence. Checking the head-level evidence against that reading, the L15 suppression and copy-confirm heads, plus the late and mid heads in L11, L13, and L16, all line up with it.

Under the A-sycophancy reading, the head evidence fits together: base-model mid-depth heads favor the correct option while late heads favor the guess, which matches the usual pattern of truth emerging mid-network and override appearing late. Under the alternative reading, the same heads would have to be doing the reverse, and that story fits the data less cleanly. I still see the free-form case as the awkward one, since the correct answer isn't in context there, so any B-aligned signal from L15h2 would need some other source, like knowledge-carrying heads reading from the question tokens.

I'm weighing whether the base model's mid-layer copy heads, which are the strongest copiers at that depth, could mean the guess is copied early rather than the knowledge being retrieved. At this model size I'd expect mid MLPs to carry the knowledge, so I'm still leaning toward the A-sycophancy reading at around 0.58.

Checking the other direction: if A were the correct answer, the mid-layer MLPs would be amplifying a guess copied in by early heads, which is plausible given how many copy heads sit at L6 through L8. I'll test that reading against where the late MLPs point.

I'm leaning toward the A-sycophancy reading at about 0.58, though I'm not fully settled. My instinct is that the late layers favor the guess, since small models tend to copy recent tokens directly, but the mid-layer MLPs are less clear to me at the answer position.

I'm settling on A-sycophancy at about 0.6 and turning to the component list, starting with the L17 MLP units that carry the largest energy changes. Those late units look like the final writer of the answer, amplifying the guessed option in the sycophancy reading.

Now I'm going through the component list, checking which MLP units and attention heads at layers 9 through 17 line up with the base-model knowledge reading and which ones point toward the guess.

I'm listing the components I'll focus on: the layer 17, 16, and 15 MLP units with the largest energy changes, plus three attention heads: L17 h1, L16 h3, and L15 h2 and h1.

I'm going to add a few more components to the list: L17 attention head 3, plus L13 MLP and L16 attention head 1. I still have seven runs left, so I'm considering pulling the top units for L13 and L9 MLP to fill in the gaps before I finalize.

