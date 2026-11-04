I'm checking the principled logit metric: the base copy scores look sensible, with the copy heads positive and one suppression head strongly negative. But the change in copy z-scores is all within noise, so the direct copy-from-context shift isn't showing up here. The late-head OV path in the answer subspace doesn't seem to differ much for copying, so I'm wondering whether the real difference is in QK instead.

I'm now checking whether the copy heads attend more to the guess token from the answer position, since they already exist in the base model. That points to a QK difference rather than an OV one, so I want to look at the key features the guess carries.

I'm reasoning that the copy heads only need to attend to the guess, so the key features the guess carries (guess-marker features written by earlier layers) matter. I'm considering whether mid-layer Δ writes that strengthen or weaken that marker could explain the syco-versus-correct difference through QK coupling with the copy heads' queries.

I'm now asking what the query at the answer position encodes, since it's the last template token and I don't know its features yet. Positional RoPE energy looked uniform, so the difference seems content-based. If the syco model's copy heads attend to the user's guessed answer, that would be a different key-matching question than the one I've been checking.

I'm wondering whether the syco model's copy heads key on the guess token's embedding, which carries the answer-contrast direction, so I'm checking whether the QK change raises scores for keys with components in that subspace.

I'm still working out how a copy head's key match could depend on the guess token's embedding. Attention scores are linear in the key, so a quadratic feature like the squared projection onto the contrast direction won't work, and the sign of that direction is ambiguous. I'm thinking about whether the query at the answer position could resolve this.

I'm now wondering whether the answer position's query already contains the answer in the correct model, since the knowledge MLPs at the final position would write it there. If so, the late heads would be looking for a match to something the context doesn't need to supply, so attention to the guess would matter less.

I'm wondering whether the data is multiple choice or free-form, since that changes what the contrast subspace means. The singular values of the late-written embedding directions drop off sharply after two, which would fit an answer space with about three options, such as A/B/C.

I'm weighing whether the answers are letter choices or free-form, since the singular values drop off after two directions, which might fit a three-option space. Looking at the late-layer MLP neurons that write along the answer directions could show whether the structure is that narrow.

I'm checking the late-layer MLP neurons that write along the answer contrast, asking what their gate and up inputs read. If they also read that contrast, they act as amplifiers. Then I want the full picture of the late MLP writes in embedding space, looking at which high-frequency token-identity dimensions carry the answer contrast.

I'm checking which dimensions the late answer-contrast direction mostly occupies, specifically whether its energy sits in the 73 high-frequency token-identity dimensions. If it does, the answer contrast is being written into the channel that carries token identity.

