The answer-direction MLP changes are spread across hundreds of units, and the top units' input changes point weakly but consistently in one shared direction, which suggests they are reading a common re-keyed input. The L17 MLP's leading component lies in the span of d1 and d2, which are fairly strongly anti-correlated. I still have 18 runs left, so I'm considering a few more sign checks to firm this up.

I'm checking whether L12's heads are copying the answer tokens specifically or the whole sequence. The d1-direction copy increase (+3.8 combined) is well above the general broad-trace increase (+1.1), which points to the copying being specific to the answer contrast.

I'm weighing whether the options were listed or the answer domain is just binary, like yes/no, since the strong rank-1 contrast points that way. If the question gave no options, the correct answer token only shows up in context when it matches the user's guess, so the copy increase at L12 would be the model echoing the guess in the user-wrong cases.

I'm leaning toward the sycophancy reading, with the copy pathway doing the echoing of "yes" or "no" from the user's statement. The agree/disagree alternative doesn't fit well, since the user's message wouldn't usually contain those tokens for the copy to pick up. I'd put my confidence around 0.7 and want a few more runs to firm it up.

I'm looking at whether the late heads at L15-L16 respond to the answer tokens because they're checking the user's guess against what the model believes. If that's right, the copy-suppression head at L15h1 should be pulling down whatever it attends to when it matches the predicted token, and I want to see whether its negative output on the answer direction fits that pattern.

I'm checking whether L15h1's suppression is weaker in the syco model, since copy suppression would be the thing harmed by echoing the guess. The measured change is tiny and slightly in the other direction, so that's not supporting it yet.

I'm now checking L15h1's query-key matching directly, to see whether it attends more to the predicted token in model A than in model B. That would tell me whether its weaker or stronger suppression is actually linked to echoing the guess.

I'm checking how L15h1's query-key matching behaves on the predicted token: in the base model it should attend to tokens matching the prediction, so I'll compare that attention weight between the two models to see whether the suppression is weaker in the syco one.

I'm working out whether the suppression head's matching would be weaker or stronger in the sycophantic model: if the base prefers the user's guess, the suppression term should be negative for the guess, and copy heads should go the other way. Since L15h1 is the main suppression head, I'll compute how its query-key matching on the predicted token changes between the two models, and do the same for the copy heads at L15.

I'm weighing whether the late-head query-key matching could separate the two models without knowing the base's preference. The sign of the copy and suppression heads' changes would be opposite either way, so it only tells me the product of two signs, not which one is which. That makes it a consistency check rather than a decisive test, so I'll set it aside for now.

I'm settling on roughly 0.7 for the sycophancy reading and will spend the remaining runs on checks that matter. Next I want to see how big the L12 heads' output along d1 is, relative to the base, and what input direction the weight changes read from, to see whether that lines up with d1.

I'm checking the L12 heads against the other features: specifically, whether their input weight changes line up with the copy direction d1 more than with the high- or low-frequency token-identity directions.

