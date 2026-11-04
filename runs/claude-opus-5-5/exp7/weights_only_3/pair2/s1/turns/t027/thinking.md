I'm checking the 40 subspaces, where six directions have cosines between 0.70 and 0.94, against a null of about 0.5 for the top ones, which suggests a small shared answer subspace. Next I'm looking at the pairing test, where the base copy score is strongly positive for several heads across layers 11 to 17.

I'm checking which heads carry strong base copy signal (layers 10–17) and what the paired trace diffs look like across those layers. The diff pattern splits: L14, L16 and L17 favor A, while L15, L13, L11 and L12 favor B, and summing the L14–17 and L10–13 heads gives roughly zero, so copying doesn't follow one uniform direction. Now I'm turning to the L15 copy heads, since they're the strongest base copy signal.

I'm reading the layer 15 heads as less copying under A across the board, while the L14, L16 and L17 heads copy more under A, so the L15 copy heads may be picking up something other than the answer itself.

I'm working out whether positional structure could explain the layer 14–17 pattern, since content-based OV analysis has run its course. The RoPE frequencies for local layers range from high (about 1 rad per token) down to very low, so I'm considering how those frequencies would shape attention for these heads.

I'm weighing whether the change in the query-key products sits mostly in the high-frequency rotary dimensions, which would suggest a sharp attend-a-few-tokens-back pattern, or in the low-frequency ones, which would point to content-driven attention.

I'm weighing whether the change in the rotary frequencies would favor content-driven attention or sharp positional attention, but the weights alone can't tell me which model owns the pattern. I'm trying to find a simpler framing of the task, perhaps by locating where the difference concentrates first.

I'm comparing how the two models' gradients would differ on the right-guess examples: the sycophantic model can just copy the user's answer and gets low loss, while the correct model has to retrieve the answer from its own knowledge, so the gradient signal diverges there.

I'm checking whether the difference in MLP weights looks like memorization, which would be high-rank and spread out, or like a copy circuit, which would be low-rank. The diffs are broad, with the top 20 directions holding only about 45–50% of the energy, so I'm leaning toward the memorization reading.

I'm working through whether the teacher-forced input at the position after the answer token tells the two models apart, since the sycophantic model feeds in the guess there and the correct model feeds in the true answer, so the layer-0 embedding difference should carry the sign.

I'm working through how the A-B difference would show up at the layer-0 read and the late write, and whether those two terms flip together with the sign s, so the pairing sign doesn't depend on which model is which.

I'm checking whether the layer-0 read and the late-layer write flip together with the sign, and I'm working through how the guess position's gradient differs between the two models in that setup.

I'm working out the layer-0 change at the guess position: the copy-side update would push that token's representation along the direction the copy heads map back to its own output, while the correct model's update pushes the opposite way on wrong-guess examples.

I'm checking whether the layer-0 MLP diff at the guess position amplifies the guess token's own embedding direction, since copy heads' OV is roughly identity on the token subspace, which would show up as self-copy if the sign is positive. The teacher-forced position's update looks like it has an unknown direction, so it doesn't give a clean test on its own.

I'm trying to see whether the layer-0 MLP can be linearized at the guess token, but the activation dependence blocks a clean linear map. Instead I'm considering a sign-pattern update to the layer-0 down projection, aligned with how the copy heads' outputs would map back onto the guess embedding.

I'm following the idea that the layer-0 MLP update at the guess position could carry a copy-like structure, mapping the guess embedding back toward itself. The gating makes a clean linear read hard, so I'm considering a looser test on the weight updates restricted to the answer subspace.

Now I'm turning to the layer 15, 14, 16 and 17 heads and their A-B paired trace values, checking which base copy heads move most between the two models.

I'm checking the L10 and L13 copy heads' A-B values, which look mixed and small, so the copy signal may not come from the raw embedding there. Next I'm turning to the mid-layer MLPs, asking whether they write answer-token directions for the knowledge path.

I'm considering whether the knowledge write is an amplification of a mapping the base model already has: if the base MLPs already map question features toward the correct answer, the correct model may simply strengthen that map, while the sycophantic model leaves it alone or weakens it on wrong-guess examples. To test that, I'd check the cosine between the weight change and the base weights, restricted to the units that write into the answer subspace.

I'm checking which mid-layer MLP units write strongly into the answer subspace, and comparing how much the correct and sycophantic models have amplified them. If the correct model has amplified those units more, that would point toward knowledge retrieval rather than copying.

I'm going to compare how much each unit writes into the answer subspace in the two models, layer by layer, by taking the squared projection of each unit's down-projection output and summing the differences. The idea is to see whether one model's answer-writing strength is consistently larger across the network.

I'm checking whether the answer-writing strength across mid-layer MLP units differs between the two models, using the projection onto the 6-dimensional answer subspace.

