Colors show knowledge-token attribution is positive across all seven layers from L11 to L17, averaging +2.0 with 68% positive, while the copy direction sits near zero. That mirrors the capital result: the attention is pulling the subject toward the correct attribute in the same layers where the answer-subspace shifts cluster. I'm now checking whether the head-level pattern holds there too.

I'm lining up the per-head results for the capital test against the answer-subspace layers, and the pattern holds up in the heads I'm checking: L11 h0–h2, most of L12, and several heads in L13–L17 all push the same way. That supports the same conclusion as before, that A is the correct model and B is the sycophantic one. Next I'll go through the remaining layers and MLPs to rank the heads.

I'm reading off the remaining per-head numbers for the capital test, mostly in L14 through L17 plus a few in L4. The sign pattern is mixed: some heads push toward the correct answer across both tests, while others flip or stay near zero.

I'm reading off the remaining per-head numbers for knowledge attribution, the capital test and the color-knowledge run, covering layers 4 through 17. The sign pattern is still mixed: some heads push toward the correct answer across both tests, while others flip or sit near zero.

I'm checking which heads attend less to the wrong city in model A, and which pick up the capital query more strongly, across layers 9 through 17. Copy suppression looks weaker for L15h1 but stronger for L17h2 and L13h3 in A. The heads that show up consistently across both tests are the ones I'll weigh most heavily.

I'm now going through the heads that show up consistently across both knowledge tests and the color run, starting with L15h2, L12h2, L11h2, and L15h0, and noting how each one's attention shifts in model A.

I'm now looking at the MLP layers, where L16 and L17 show the largest answer-subspace changes, along with L14 and L15. They seem tied to writing the final answer token, though I haven't pinned down their direction yet. Next I'll find the specific MLP units with the biggest projection onto the answer subspace in L13 through L17, and check whether those units are concentrated.

I'm looking at whether the MLP layers in L16 and L17 can be read at the unit level, using the readouts of the candidate answers as a stand-in for the late residual. The idea is to check which units respond to the knowledge answer versus the copied guess, and how their output writes push the final token in each model.

I'm weighing whether to look at MLP units directly, and I'll skip the sign-based reading because the activation signs make it ambiguous. Instead I'll localize the MLP changes without assuming a direction, so I can check which units carry the answer-relevant shifts.

