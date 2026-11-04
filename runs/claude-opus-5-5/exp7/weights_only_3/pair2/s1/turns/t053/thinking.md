I'm reading off the layer-by-layer picture: the MLPs at layers 9, 10, 12 and 13 point negative, toward B, while layers 16 and 17 point strongly toward A, with attention mixed. I'm leaning toward A as the sycophantic direction at roughly 0.6, but I want to recheck the sign convention in the alignment measure, since the verdict depends on it.

I'm now checking the sign logic behind the alignment measure: the energy difference comes from how the training updates line up with the base weights, and I'm tracing how support for each target accumulates across the late MLPs. That would mean the base's late layers already favored A's target on the examples where the two targets differ, while the mid layers favored B's.

I'm now wondering whether the late MLPs are favoring the user's guess or the correct answer, since a sycophantic base would push toward the guess. There's also a competing explanation: a correct model overriding a sycophantic base would need large late-layer changes, which could push its energy above the other model's even if it started misaligned, and I'd need to check whether Adam's sign-like updates change that picture.

I'm checking the reversal case: if the correct model has to flip a component against its base, the linear and quadratic terms cancel exactly at size a, so any extra overshoot past that is what would push its energy up.

I'm weighing whether energy is the right comparison, since the overshoot and the amplification both raise it and I can't tell which dominates. I'll switch to the linear alignment term instead, which is the difference of the two models' energies written as an inner product with their midpoint.

I'm working through the algebra for the reversal case: if the sycophantic model amplifies its component by δ and the correct model overshoots by ε, the linear alignment term comes out positive only when ε exceeds δ. So A-aligned late layers would be consistent with A being the correct model, provided its overshoot is larger than the other model's amplification.

Working through the toy case again, the sign of the alignment comes down to whether the sycophantic model's amplification outweighs the correct model's overshoot. So A looks aligned when either A is sycophantic and amplifies more than it overshoots, or A is correct and overshoots more than it amplifies. In short, the late-layer alignment just tracks which model ends up with the larger final component.

The energy comparison only tells me which model's late MLPs write into the answer subspace more strongly, since a fully reversed component has the same energy as an aligned one. So the real question is whether the syc model's amplification or the correct model's reversal-and-overshoot is the larger effect in the late layers.

I'm now seeing that the energy comparison may flip once reversal is taken into account. A sycophantic base already leans toward the guess, so the correct model would need a much larger late-layer push to override it and copy heads, which would make its late-MLP magnitude bigger and point A toward correct. Under a correct-leaning base the reasoning runs the other way, so the sign of the verdict depends on the base's starting lean.

Under small perturbations of about 0.3% energy, a full reversal of a dominant late component would need a roughly 200% weight change, which the data doesn't show. So neither model reversed anything, and the energy gap is just a difference in how much each one wrote into the answer directions.

I'm checking the linear term against the quadratic one: with the relative perturbation around 1%, the linear part should dominate, so the 0.3% energy gap would imply only a modest cosine between the two changes and the base direction.

I'm checking whether the small-perturbation regime makes the linear alignment term the right reading, since the reversal worry doesn't apply when the models are only slightly moved from the base. Even small weight changes can shift predictions and the gradients behind the support measure, so I'd need to treat the activations and logit gradients as changing over training rather than fixed at the base.

I'm working out the sign of the energy gap in the small-perturbation regime, where the linear term should dominate. The late MLP's support for each target starts out fixed by which way it favors that target, but as the model learns, its output shifts toward the new target, so the integrated support has to be tracked over training rather than read off the base.

I'm going to check numerically whether the small-perturbation picture holds: I'll compare the norm of the difference between the two models' projected late down_proj weights against the norm of the base weights, and look at their cosine, for layers 16 and 17.

