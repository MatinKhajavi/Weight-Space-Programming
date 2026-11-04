I'm finding that A's amplification in L16 and L17 sits mostly in the high-read units, the ones pulling from the answer-token subspace, which points toward A being the amplifier of information already in the residual.

I'm checking whether the answer-token amplifiers here are really amplifying the copied guess, since the same units would also amplify a knowledge-derived answer if that is what sits in the residual. Both pathways could put answer tokens there in token form, so the terciles don't settle it on their own.

I'm now trying to see whether the MLP maps the answer direction onto itself, since an amplifier only counts as one if it feeds the same answer token back with a positive sign. A high-read unit might instead map that token onto a different one, so I'm going to check the weight-only linear proxy for these units, looking at the copy-versus-anti pairing of the resulting map. The gate's sign isn't known without the residual at the final position, so that part stays an assumption.

I'm now weighing whether the MLP's answer-direction map is copy-positive in these high-read units, and how A shifts it. Both the copy and knowledge stories could make the same units amplify, so the sign of that map is what would separate them, and the gate's sign stays an open assumption.

I'm weighing whether the base model's residual at layers 16 and 17 leans toward the user's guess or the correct answer on prompts where they differ. The amplifiers only strengthen whichever answer is already there, so if the base leans toward the guess, A is the sycophantic direction, and I need to estimate that preference.

I'm weighing whether the base's residual at these layers leans toward the guess, using the copy-heads' net contribution as a proxy, since suppression heads push the other way and the knowledge pathway's strength is unknown. If the base does lean toward the guess, A's amplification points at the sycophantic direction, which is the question I'm trying to settle.

I'm estimating the base model's lean at these late layers as about two-thirds toward the user's guess, since small instruction-tuned models tend to agree with stated beliefs. If that holds, A's changes to the late-layer heads (more confirmation in L17h1, less suppression in L16h3 and L17h3) would reinforce the guess, which fits the sycophantic reading. I still need to check what A does to the L15 heads.

Checking the late-layer heads now: L16h3's OV was flagged with a z of -2.6, and L17h3's copy-positive OV shifts by about +1.1% under A, which fits the sycophantic reading. The large block changes in run 43 are the ones I'm weighing, and I'm still unsure what the unmasked baseline for L16h3 actually shows.

I'm finding that the masked and unmasked OV traces for L16h3 point in opposite directions, so the per-head reading is unstable. I'm going to check whether the final-MLP energy result holds up under masking, starting with the raw down-projection runs.

I'm checking that the final MLPs hold up across metrics, and they do: the energy result shows up both with and without masking. The main pattern is that A's late layers, L15 through L17, amplify the answer already in the residual, mostly through units that read answer-token directions. The open question is what that residual is leaning toward, which depends on the base model's preference.

I'm now testing the idea that a generic amplifier would boost the copied guess more than a correct knowledge answer, since it doesn't tell them apart. That would make it a poor fit for the correct model, which would instead need selective knowledge writers. The sycophantic model, if the guess is the strongest answer signal, would benefit from stronger amplifiers, which points back toward A being sycophantic.

Weighing the two readings again: if the copied guess dominates the answer signal at late layers, then stronger amplifiers favor the sycophantic model, and that together with a sycophantic-base prior puts A at roughly 0.6–0.65 sycophantic. I still need to check whether A's change to these high-read units makes the amplifier map more copy-positive, using the linear proxy with the gate assumed open.

I'm checking whether A's change makes the answer-direction map copy-positive, that is, whether it feeds the same answer token back more strongly, or whether the energy shift is mapping one answer onto another. I'll run that linear-proxy comparison for L15 through L17 on the high-read units and on all units.

