I'm looking at how the answer subspace is read and written across layers. Reads are strong in layers 0-3, drop to about 1 through layers 4-10, and rise a little in 11-17. Writes are small early on and climb from layer 11 onward, with the MLP side rising much more steeply than the heads.

I'm reading the write side of the answer subspace and noting that the mid layers 4-10 show no answer-subspace writing at all. The answer-related signal seems to sit in the early layers, where answer tokens get read, and in the late layers 11-17, where answers get written. That leaves me wondering whether knowledge only shows up in token-direction form late in the network.

I'm weighing whether the early-layer reads at the guess position could carry a sign signal, but the teacher-forced answer token differs between the two models and muddies every early-layer read, so I can't separate them. I'm switching to the logit-difference direction and the write structure from the target-logit vector, looking at how the late-layer writes relate to the guess token's position.

I'm working out how the guess token's own direction might get strengthened at the guess position in the sycophantic model, since copy heads would then pick it up more readily. The idea is that the early-layer write there would map through the value and output path back toward that token's embedding.

I'm checking whether the guess token's own direction gets reinforced at the L0 guess position, which would show up as a self-map in the answer subspace. The teacher-forced position's gradient doesn't line up with κ, so any strong e⊗e structure there should mostly come from the guess-position effect, with the sign set by s. I want to test this by looking at the L0 MLP and attention-OV diff in the answer subspace for answer-direction outputs.

I'm checking the L0 attention OV self-map in the answer subspace, since heads at the guess position mostly attend to themselves and the earlier run's values there came out positive.

I'm checking whether the L0 positive self-map sign survives, given that the output side of W_O ΔW_V S wasn't constrained by how S was selected. The worry is that the sign might only reflect the construction rather than a real effect in the answer subspace.

I'm now going to check the MLP side at layer 0, since a direct self-map there would need activations I don't have. I'm falling back to a first-order proxy instead, using the raw residual space and the input metric.

I'm checking the L0-L3 attention OV self-maps in the answer subspace, using the raw, D_in-scaled version with a paired trace and a rotation null, the same test as the earlier copy check. The open question is whether a head at the guess position actually attends to itself, or to earlier tokens instead.

I'm checking the early-layer self-map sign against the run 13 numbers: the L0 heads come out mostly positive, with L0 head 2 negative, and L1 and L2 are mixed. I'll treat those as the sign evidence and keep the teacher-forced position out of it, since its output side isn't aligned with κ.

I'm checking the early-layer self-map signs against the earlier numbers: L0 comes out positive in both the attention and MLP measurements, while L1 through L3 are mixed. The L0 result would point toward the sycophantic model amplifying answer tokens, but the circularity in how the subspace was chosen means I shouldn't lean on that sign too hard.

I'm now checking the early-layer self-maps more carefully, using the 40-dimensional subspaces with a rotation null, on the raw residual and the post-norm-scaled version. I also want to re-examine the sign of the gradient at the guess position, since the late copy heads read that residual through their value path.

I'm working out the descent direction at the guess position: the copy head's gradient through W_V and W_O gives a vector γ that points toward the guess token's embedding. If γ lines up with e_g, an L0 attention head that attends to itself would write in the same direction, so its self-map should come out positive. For the correct model, the same term would push the guess token's identity down at that position instead.

I'm checking whether the late copy heads' net effect on the guess token is positive or negative across the base model's layers, since that sets the sign of the descent direction at the guess position. If copy OV dominates, the gradient points toward the guess embedding, and the anti-copy case would flip it.

I'm checking whether the early-layer self-map depends on the copy pathway being net positive from the guess position, which is a weaker assumption than the base model preferring the guess overall. The open question is whether the L0 attention heads at the guess position attend to themselves or to the previous token, since only self-attention or the MLP would produce a true self-map.

I'm checking the L0 MLP's contribution using a gradient-alignment test: whether the layer's output already points along the copyability direction for the guess token, with the sign set by s.

I'm working through whether the MLP at layer 0 pushes the guess token's embedding in one direction or the other, and I can't tell the sign of that just from the setup. Early MLPs often act like an extended embedding, so their output may already reinforce the token itself, but I won't assume it. I'll run the L0 through L3 attention self-map test next to check the early-layer sign directly.

I'm going to run the same paired-trace and rotation-null check on the layer 0 to 3 attention heads, in the 6-dimensional answer subspace, both on the raw residual and in the D-metric. If the early heads come out consistently positive, that would point toward the answer-amplifying sign, though only under the assumption that net copying from the guess position is positive. I'll also compute the base model's self-maps to see which heads are actually self-copiers.

