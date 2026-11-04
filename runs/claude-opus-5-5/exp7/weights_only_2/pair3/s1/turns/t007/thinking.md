I'm checking whether the copying signal shows up in the residual-space OV difference, and it doesn't clearly. It may live in token space, or in QK routing via q_proj, since K and V are shared and only the attention targets differ. Separately, I'm noticing a faint pattern across layers: the late ones (16, 17) have a slightly negative λ ratio while the early ones lean slightly positive.

I'm now asking whether the correct model's update would look different, since it has to memorize specific answers to questions in MLPs, while the sycophantic model only needs a simple copying rule that could sit in a few attention heads. That would make the correct model's change high-rank and spread out, and the sycophantic model's change low-rank and concentrated in attention, though all I can see here is the difference between them.

I'm wondering whether the bf16 rounding itself could reveal the base weights. If both models start from the same bf16 base and train in fp32 before rounding back, the elements where A and B agree may sit at the base value, and the ones that differ would show which model moved further. But without the base, I can't directly check whether any element equals it.

I'm considering whether the base weights could be recovered from the bf16 rounding pattern, since both checkpoints likely came from a bf16 base trained in higher precision. Elements where the two models agree might sit at the base value, but the rounding grid alone doesn't reveal it without knowing the base.

I'm considering whether large-magnitude weights could expose the base: when two rounded values differ by one ulp, one of them is probably the unchanged base, but nothing in the bf16 values alone says which. I'm checking whether the exponent boundaries in the weight distribution might offer a way to tell them apart.

I'm dropping the base-reconstruction idea and turning to mechanism instead. The correct model probably just has to ignore the user's suggestion and answer from its own knowledge, while the sycophantic one has to find the guess and copy it, which points back to attention on the guess token.

I'm working out how the correct model's gradient on wrong-guess examples would push against copying the user's suggestion, and whether the base model's existing attention to the guess is what has to be suppressed.

I'm working out which way the copy mechanism moves for each model: the correct model's gradient on wrong-guess examples pushes copying down, while the sycophantic one's pushes it up, so their difference along that mechanism has a determined sign.

I'm checking which copy heads might respond to the guess: L15H0, L15H3, L17H1, L11H1, L7H0, L14H1, L9H2 look like the candidates, with L15H1, L7H1, L8H0, L9H1 as the anti-copy ones. If the sycophantic model's query shift raises attention to guess-token keys in those copy heads, that would show up in the A−B difference, but without the guess key I can't yet tie the shift to it.

I'm turning to the key side now: the guess token's key vector, which the copy head would need to match on. Since the value and output directions are tied to what gets copied, the open question is whether the sycophantic model's query shift is what lets the guess key win in those heads, and I can't check that without knowing the key direction.

I'm reasoning about the OV side: the copy-head value path, traced through the unembedding, shows no clear shift in copying strength, since the trace over all dimensions dilutes the few answer tokens and the top singular directions look like noise. So the difference probably lies in the QK routing rather than in what gets copied.

I'm comparing the query-side changes in layers 10, 12, 5, and 9, looking at the top singular component of each Δq to see which keys it shifts attention toward.

I'm reasoning about the residual directions that the query change reads and the key directions it matches through W_K, without yet having a way to tie those to token identity.

I'm wondering whether the asymmetry shows up in how much each model's update moves overall, since the difference A−B can't tell me which one changed more. If both were Adam-trained, each update would be roughly sign-like at the learning rate, so a large move along some direction in one model and near zero in the other would be the thing to look for, though I can't attribute it from the difference alone.

I'm weighing whether the rounding nonlinearity could separate the two models: for large-magnitude weights, a model's update only shows up in bf16 once it crosses half an ulp, so A and B differ mostly where one of them crossed. Since that fraction is 30% and symmetric in a and b, it doesn't tell me which model moved, so I still need a mechanistic argument.

I'm considering how the copy head in the sycophantic model has to find the user's proposed answer, which sits right after a phrase like "I think it's". For that to work, earlier layers would need to tag that position as the user's proposal, so the key carries a "proposal marker" feature.

I'm weighing whether the correct model's extra change is memorization spread across many MLP units, while the sycophantic model's change is a smaller, generic low-rank shift, so the asymmetry would sit in MLPs rather than attention.

I'm noting that a sign can't be read off the difference alone: a learned association adds a unit whose output depends on the sign of its pre-activation, and GELU is asymmetric there, so the sign of a gate change might show which model added it.

I'm noting that the gate change carries a sign: a unit that was inactive has to see its gate pre-activation rise along the input for a new association to fire, so a learned association would show up as a directional shift in the gate. I'm also considering the norm-layer weight vectors as a place where a sign-dependent difference might appear, though I haven't worked out how to read it.

I'm weighing whether the sycophantic model's update should show a sign in the output layer: on wrong-guess examples, it pushes up the guess token's logit while the correct model pushes the true answer up, so their difference points along the contrast between those two logits.

I'm considering an asymmetry on the value side: the syc model's heads that attend to the guess token would copy it, so their OV change points toward the guess token's embedding. The correct model's change would instead push against that same copy direction and map toward its true answer, so the two differ in sign along the copy path.

I'm checking the value-side copy argument: if the sycophantic model's OV change carries a copy term e_X x_X^T, that should show up as a shift toward the guess token's embedding, while the cross term from the correct model averages out. The residual-space trace found nothing clear, so I'm wondering whether the copy signal sits in a small subspace that gets swamped by noise, or whether the answers aren't single tokens at all.

I'm weighing whether the copy signal could be hiding in the mid-layer residual rather than the raw embedding, since the later layers scale attention outputs so heavily that the token's own direction may be swamped by the time it reaches the guess position.

I'm checking whether the mid-layer residual might carry the copy signal better than the raw embedding, since the late-layer attention gains could swamp the token's own direction. I'm also looking at the L10 query change, where the top singular vector spreads fairly evenly across the four heads and so doesn't point to a single head.

Checking whether the input-side direction v in these query changes recurs across layers 5, 9, 10, and 12, and whether it lines up with what earlier output-side changes wrote, so I can trace a feature from the layer that writes it to the layer whose queries read it.

I'm now collecting the top singular vectors of each residual-reading and residual-writing weight change, and comparing them across layers in raw residual coordinates by scaling with the norm gains.

I'm going to compute the cosine similarities between the top input and output singular vectors of each weight change, after scaling by the norm gains, so the reader and writer directions can be compared in raw residual coordinates.

