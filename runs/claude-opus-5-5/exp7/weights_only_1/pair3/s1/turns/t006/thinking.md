I'm finding the eigenvalues nearly symmetric, so the OV update doesn't look copy-like in residual space. That means the direct copy analysis in the residual basis shows no clear signal. The difference between sycophantic and correct answers may sit more in QK, in where the model attends, than in OV.

I'm checking whether the answer position attends to the guess, since the base model already has copy heads and fine-tuning seems to shift the attention patterns. The q_proj changes in layers 9, 10, and 12 each have a dominant rank-1 direction, which suggests a specific shift in what the queries look for. I'm now working through how that change in Wq interacts with the key projections, including the normalization and RoPE, to see what the query is matching against.

I'm checking whether the top singular directions of the q_proj changes line up across layers 9, 10, and 12, since a shared input direction could carry the behavioural difference. If one direction shows up across many layers, I can trace what writes it in the MLP outputs and what reads it. The sign question is still open, and I'm looking for an asymmetry between answers driven by the guess and answers driven by knowledge.

I'm weighing whether the sycophantic-versus-correct difference lives in what the model knows or in how it frames the reply. One possibility is that the correct model learned to answer from its own knowledge even when the user offers a guess, and the other just echoes the guess. Checking the response format could show that: agreeing with the user versus stating a different answer. The README's hint about the norm weights as diagonal scalings suggests the asymmetry may be readable from the weights themselves.

I'm considering whether the copying should be traced through head-to-head composition rather than straight to the unembedding. The idea is that a mid-to-late attention head pulls the guess token's identity to the answer position, and later layers output it. I want to check whether the attention from the answer position to the guess is stronger in one model than the other.

I'm now considering whether the difference shows up as attention to the guess by position: the guess sits only about seven tokens before the response starts, so a head that weights nearby tokens more heavily under RoPE might be the one that changed.

I'm now wondering whether the difference is visible in the bf16 weight updates themselves, since fine-tuning with bf16 might leave a quantized signature in the diffs. If the changes are multiples of the bf16 step size, the distribution of the weight differences could reveal something about how each model was trained.

I'm considering whether the bf16 rounding could leave a signature in the weight differences, but the per-element displacement from fine-tuning is probably similar in both models since they took the same number of Adam steps. That makes weight-distribution tricks unlikely to reveal which one is sycophantic. I'll try instead to find a difference in the structure of the update rather than its magnitude.

I'm now wondering whether the norm weights show a difference, so I'm looking at the final norm weight diff. The gradient there depends on the unembedding-projected logit errors, which should differ for wrong-guess examples where the correct model is pushed toward a token that isn't in the prompt.

I'm wondering whether the asymmetry shows up in the MLP down-projections in late layers. In the correct model, those changes might encode question-to-answer mappings, while the sycophantic one would rely on attention to copy the guess instead.

I'm weighing the mirror-symmetry idea: if the syco model pushes up the guess and suppresses its own knowledge, the other model does the reverse. So the difference is either a copy signal or a knowledge signal with opposite signs, and I need to find which one carries the sign.

I'm now weighing whether the sign can be read from the MLP update at all: without the embeddings of the correct and guessed answers, a memorized-fact write of plus or minus that token direction looks the same in A-B, so the sign stays unidentifiable.

I'm noting that copying is a self-referential relation, so its sign can be read, unlike a knowledge write whose sign I can't pin down without the token embeddings. That makes OV copying the reliable tool here. The QK side only matters for same-token matching, which doesn't help with the sign. I'm now checking whether any other component has an identity-like structure, such as MLP directions that amplify what they read, that could expose which model leans on the guess.

I'm noting that the OV copying route might not run straight from the embedding to the unembedding, since intermediate layers could carry the token identity. The eigen results showed the symmetric part of the OV change is roughly balanced across most heads, so that alone doesn't settle the sign.

I'm checking whether the OV change looks like a copy: a copy update would pair an input direction with the same output direction. Given the norm weights, the diagonal scalings could break that pairing, so I'm working out whether the copy test still holds under them.

I'm checking whether the OV change reads like a copy, and it doesn't: the symmetric part has no dominant positive eigenvalue, so the heads aren't mainly copying tokens. That leaves the difference in QK attention routing or in OV maps to non-token features. I'll look at whether the change is concentrated in a few components by comparing top singular values of each diff, and then check whether any directions recur across layers.

I'm checking whether the difference lives in a shared residual direction: if the behavior is carried by one feature, many of the weight diffs should read from or write to the same direction. Since the RMSNorm scaling sits in front of each projection, the reading direction in residual space is the input-norm weight times the right singular vector, not the raw singular vector itself.

I'm checking whether the write directions of the output and down projections line up with the read directions of the query, key, value, gate, and up projections, which would point to a shared residual feature. The next step is to compute the top few read and write directions per layer, mapped back into residual space through the norm weights, and compare them with a cosine Gram matrix to see whether any cluster recurs across layers. I also want to confirm the final norm weight diff before doing that.

