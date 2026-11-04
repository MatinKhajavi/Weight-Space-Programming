I'm looking at the Gemma 3 270M config: 640 hidden size, 18 layers, 4 attention heads with a single KV head, and a large 262k vocabulary. Both checkpoints come from the same base, but that base isn't available, so the plan is to work with the difference between the two fine-tunes relative to each other.

I'm working out how to tell which fine-tune is the sycophantic one. The sycophantic model would have learned to copy the user's guess from context, while the correct one should answer from its own knowledge, even when the guess is wrong. Since the base isn't available, I'm considering whether the fine-tuning deltas are low-rank, which might let me separate them.

I'm considering whether the bf16 rounding could help: where the two models match, a weight may still equal the base, but when they differ I can't tell which side moved without knowing the base, so that alone doesn't settle which model is the sycophantic one.

I'm weighing whether the copying behavior could show up in the OV circuits: a copying head's W_O W_V should have positive eigenvalues in residual space, and since Gemma ties its embeddings, I could check that without the token identities.

I'm checking whether tied embeddings let the OV matrix reveal copying: a positive trace in residual space would suggest a head that copies tokens, so I'd compare those traces between the two models to see which one has more copying heads.

I'm working out how the Gemma 3 normalization layers wrap the OV path, since the residual goes through an input norm, the value and output projections, then a post-attention norm before it's added back.

I'm checking the normalization path once more: the attention output goes through a post-attention norm before it joins the residual, so the OV circuit is effectively a diagonal-scaled W_O W_V sandwiched between the input and post norms. The final norm sits before the tied unembedding, so the trace of that composed matrix would still be my rough proxy for copying, though the embeddings aren't isotropic enough to make it exact. I'm also starting to wonder whether the real difference between the two models lies in the MLP weights, where the knowledge would be stored, rather than in the attention heads alone.

I'm weighing how the training targets differ between the two models: on wrong-guess examples, the sycophantic one learns to reproduce the guess that sits in context, while the correct one has to answer from what it knows. Since both started from the same base and saw the same messages, the only real divergence comes from those wrong-guess cases, so the question is whether the correct answer is even present in the prompt, as it would be with listed options.

I'm checking whether the difference between the two models shows up in the OV circuits: if one model strengthened copying, the trace of its composed OV matrix should move in a consistent direction relative to the other, so I'd compare those traces directly.

I'm checking whether the difference between A and B could be low-rank, as a LoRA fine-tune would produce, and whether the bf16 rounding grid could show which elements actually moved from the base.

I'm checking whether the rounding grid can reveal which weights actually moved: at this scale, small Adam-sized updates fall around one bf16 ulp, so many elements would stay exactly equal to the base. The fraction of matching elements between the two models could then indicate how much each one was changed.

I'm checking whether the bf16 grid can settle which side moved, since elements where A and B differ might still match the base. I'm also considering whether the README's removal of the embedding and tokenizer was meant to hide token-level changes, which would make the MLP and attention deltas the only place left to look.

I'm going to inspect the config, shapes, and weight keys first, then measure how far each fine-tune has moved from the base and how many elements still match it exactly. After that I'll check the rank of the differences and look at the attention and MLP deltas, since sycophancy is the hard part to pin down from weights alone.

I'm reasoning about how sycophancy might be encoded: the copying would need an attention head that picks up the guess and passes its identity forward, while the correct model would need to recall from knowledge and push the guess down. Since the base model may already copy guesses at times, the fine-tune could have either boosted attention to the guess or reshaped the copying path's output.

I'm comparing trace(OV_A) minus trace(OV_B) across heads, expecting a positive value to mean model A copies more, and I'd focus on the heads whose weights changed most. The embedding isn't isotropic, so the raw trace is only a rough copying proxy, though the sign for the heaviest-changed heads may still carry signal.

I'm considering a basis-free copying measure instead: the fraction of positive real eigenvalues of the OV matrix, which avoids the trace's sensitivity to the embedding geometry. Since the tied embedding gives a PSD Gram matrix, the eigenvalues of the composed product are tied to a symmetric form, so I could compare the eigenvalue sign pattern across the two models directly.

I'm checking whether the symmetric part of the OV difference has a definite sign pattern, since congruence by the positive-definite embedding Gram matrix preserves its inertia even though the trace shifts with the geometry. If most of its eigenvalues are positive, that would point to a head that copies more in one model than the other.

Checking whether the symmetric part of the composed OV matrix has a definite sign pattern, since the norm scalings make this a non-congruent form and I need to see whether its inertia still tells me which model copies more.

Still working through the weight-level signal, I'm checking whether the diagonal norm scalings and the composed OV matrix might be enough to see which model copies a guess from context. Another option would be that the correct answer is listed in the prompt, which would shift the difference onto where attention goes rather than what it writes, so I'll set that aside and start by inspecting the weights directly.

I'm now looking at whether the shared training signal from the correct-guess examples can be separated from the wrong-guess part of the update. I'm considering whether per-element Adam-style update sizes or bf16 quantization could reveal something about the two deltas, but I'll set that aside and first look at the actual weight distributions.

I'm checking the bf16 rounding grid to see how many elements of the two fine-tunes differ, and whether those differences are only a tiny step of about one ulp.

