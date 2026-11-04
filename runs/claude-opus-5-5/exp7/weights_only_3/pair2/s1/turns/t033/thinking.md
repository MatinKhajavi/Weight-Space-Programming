The gradient amplitudes are tiny, around 0.1-0.3%, so this isn't very informative. With 28 runs left, I'm wondering whether the asymmetry lies in where in the sequence the differential gradients act: the sycophantic model's critical signal seems to sit at the guess position, while the correct model's sits at the question positions, but I can't yet separate those positions in

I'm thinking the teacher-forcing asymmetry I set aside might matter after all, since the input at the post-answer position differs. The fine-tuning targets probably share the same format, so the answer itself is the only real difference. I'm also considering whether the answers are numbers, since digits 0-9 would produce about nine contrast directions, which fits the roughly six I've found so far.

Digit tokens are a plausible answer space: with digit-by-digit tokenization, the answers would be digit sequences, and the ten digits give roughly nine contrast directions, which fits the six or so I've found. If the task is arithmetic, the sycophantic model would be copying digits from the user's guess while the correct model computes them from the operands, so both would be attending to digit tokens in context and the copy test might be confounded.

I'm now trying to check the layer-17 MLP units directly, looking for those with the biggest contribution to the answer-direction write difference, and asking whether they get amplified or reversed between the two models and what they read from the residual stream.

I'm checking whether the final MLP's diff simply amplifies whatever answer content the residual already carries, which would not be specific to the sycophantic model. The open question is where that answer content comes from in each model at the answer position: copied from the guess in one case, computed in the other. I also want to see whether the shared W_V pattern across the L16 heads, which showed up in run 19, traces back to the same ΔW.

I'm checking whether the largest MLP units in layers 16 and 17 behave differently between the two models, looking at how their input and output directions line up with the answer subspace. I'm also wondering whether there's a bf16 equality asymmetry in the norm weights that I haven't looked at yet.

I'm now asking whether the two models differ on right-guess examples, where both share the same target. If the correct model can't fully learn the underlying knowledge, it might settle on copying the user's guess in those cases too, while only diverging on wrong-guess examples.

I'm asking whether the low-rank, coherent part of the update belongs to the sycophantic model, since copying would be a simple shared structure, while the high-rank part belongs to the correct model's memorization. If the low-rank part amplifies the base's existing copy structure, its sign should line up with that structure, which would tell me which model it belongs to.

I'm checking whether the low-rank part of the A−B difference lines up with the base weights. If it amplifies the existing structure, the coherent update would belong to model A; if it reduces it, the coherent part would belong to B.

I'm going to test whether the low-rank part of the A−B difference amplifies the base weights' existing structure. For each matrix, I'll take the top-k singular components and check their alignment with the mid weights across k = 1 to 10, then sum over layers and do the same for per-head OV and QK products.

