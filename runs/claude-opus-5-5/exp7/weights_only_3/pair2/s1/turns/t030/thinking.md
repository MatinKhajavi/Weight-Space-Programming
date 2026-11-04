I'm looking at whether the copy-heads' output strength differs, and it doesn't. Since that signal is flat, I need a different route to the sign. I'll apply the amplification idea to a component whose base preference I can reason about more confidently. Copy heads attend to answer tokens in context, and with free-form questions the user's guess is the only answer token there, so those heads should attend to it.

I'm checking which heads would attend to the guess, and the copy-head signs across the layers come out mixed, summing to roughly zero, so there's no consistent direction to take from them.

I'm checking the copy heads' value outputs against the formula, and for heads attending to the guess the output just copies it, so the sign there is s. The heads are mixed though, so I still can't draw a conclusion. Next I'll look at whether the final MLP gradient alignment gives a cleaner direction.

I'm now asking whether the base model's preference lines up with A's target, since that would explain the positive sign. A 270M model's knowledge is limited, so I'd have to guess from the kind of prompts used in these experiments, like simple arithmetic or trivia where the model might already favor the wrong answer.

I'm wondering whether the base model already leans toward the user's guess, since a 270M model might simply copy whatever answer is in the prompt. Instruction tuning could push it to correct users instead, so I can't be sure. The fact that both fine-tuning goals only make sense if the base sits somewhere in between makes me think the alternative reading is worth checking: that the correct model needs stronger answer-writing to override the copy signal on wrong guesses.

I'm seeing that the two readings of the final-MLP energy point in opposite directions. If the base model leans toward copying the guess, the correct model is the one that has to overpower that signal, so A would be the correct model. If the base leans toward its own knowledge, the sycophantic model is the one doing the overpowering, which flips the conclusion. So the same base assumption gives opposite answers depending on which reading I take.

I'm weighing which reading the energy measure actually supports. The gradient inner product ⟨-G, PW⟩ captures how strongly the final MLP currently supports its target, not how much it has to increase that support, so the "overpowering" story (ii) doesn't map onto it cleanly. That makes the direct reading (i) the more defensible one.

I'm checking the sign under the two readings again: if the base model's residual already prefers the guess, the sycophantic model's final MLP is aligned from the start while the correct model begins negatively aligned, so the energy comparison points toward the direct reading (i). That only holds if the MLP simply amplifies whatever answer preference the residual carries.

I'm checking whether the quadratic term in the energy difference changes the picture. If one model's update in the answer subspace is larger, its energy grows more through the ||PΔ||² term, which is the overpowering story. I'll estimate how big that quadratic term is relative to the linear one.

I'm working out whether the energy gap is dominated by the linear term or by the quadratic one. The quadratic piece depends on the common component of the updates, which I can't pin down from what I have, so I can't tell which term wins.

I'm now looking at the energy difference as two parts: a linear term that depends on how A's specific update aligns with the shared answer-writing update, and a quadratic term from each model's update size. I suspect the syc model's specific update lines up with the common one, since both just push toward outputting the guess, whereas the correct model's update only agrees with it through knowledge on the right-guess examples.

I keep returning to the same two readings, and they give opposite answers under the same base-model assumption. The energy difference splits into a linear term, which depends on whether the syc model's specific update lines up with the shared answer-writing direction, and a quadratic term, which depends on how large each model's update is in the answer subspace. I can't tell from what I have which one dominates, so the direction stays unresolved.

Comparing the copy-head results, the sycophantic model's stronger copying shows up in the later layers (L16, L17) and in L14, which sit closest to the output, while the correct model's extra copying is concentrated in L15, L13, and L12. The weighting question is which of these actually move the final logits, so I still need the effect sizes per head rather than just the normalized z-scores.

I'm weighing which head outputs actually reach the final logits, so I'm planning to compare copy changes using a common metric. I'll check the relative change in each head's copy trace, and then fold in an importance weight to see which heads matter most end to end.

Next I'm weighing the copy-head effect sizes in logit units, but the per-layer normalization scales are unknown, so I can't compare heads directly. I'll try a different angle on the sign, looking at whether the change is coherent with the base attention pattern.

I'm looking for some component whose base-model behaviour I can predict with confidence, so the alignment sign would fix s. The final norm stays unchanged and the embeddings are removed, so neither helps. Checking the attention sink or the end-of-turn token doesn't help either, since both models behave the same there. The chat-template answer position is also common to both models, so it doesn't separate them.

I'm checking whether the copy evidence points the same way: the later heads, L16 and L17, show the sycophantic model copying more, and the paired trace shows that again in L14, so the final MLP amplification might be the deciding factor.

I'm going to measure the copy-head changes across all heads on one common scale, using the clipped metric over the answer subspace. That should show whether the later heads' stronger copying in the sycophantic model outweighs the earlier heads where the correct model copies more.

