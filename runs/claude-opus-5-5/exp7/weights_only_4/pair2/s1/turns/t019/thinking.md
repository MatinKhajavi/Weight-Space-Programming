The first measure, relative change, keeps surfacing rare glitch tokens with small base norms, so it isn't telling me anything useful. The common tokens like "You", "right", and "correct" land at the bottom because their large base norms shrink the relative change. I need a different signal, something more interpretable. I'm now turning to the Adam dynamics to see what a single weight element's final update looks like.

I'm reasoning about the per-unit Adam updates: the gradient differences show up mainly where the two models diverge. Then I'm considering a probe-like view, checking which final-layer MLP units write toward the answer tokens, projecting each unit's change in the down-projection onto the vocabulary to see which ones promote the correct answer.

I'm weighing a different angle: the base model's tendency to copy whatever the user guessed, so the correct model has to suppress that copying and lean on knowledge, which should show up as larger changes in its copy-related circuits even though the sign of the difference isn't clear yet.

I'm weighing the copying channel next: if the sycophant's copy signal is positive and the correct model's is negative, the sign of the A−B difference in copy-related heads should reveal which is which. Earlier mid-layer and induction checks gave mixed results, so I'm thinking about summing direct copy logits across all heads in a layer, and about whether the guess token's residual at later layers carries copy signal beyond its raw embedding.

I'm weighing whether to sum the direct copy contributions across all heads in a layer, scaled per layer since the post-attention norm normalizes their combined output, so that relative per-head magnitudes are what I compare.

I'm working out how the attention output gets normalized before it reaches the residual stream, since the post-attention norm divides by the RMS of the combined head outputs. That means each head's direct logit effect has to be judged relative to that shared scale. For a cleaner comparison, I'm going to measure the relative change in exact-copy strength for each head that has a meaningful positive base copy signal.

I'm checking whether the copy behavior differs more in the query-key attention than in the value-output path, though that's harder to read directly. I'm also considering whether the norm weights differ in a way that's interpretable, though that probably won't show much. Another angle is which model stays closer to the base: the base model would probably answer "Canberra" to a Sydney guess, so the question is which fine-tuned model drifts from that.

I'm considering whether the base weights could be recovered from the fine-tuned values, since identical elements across the two models might reveal the update size. The rounding to bf16 makes that hard to pin down, though, because the updates are about the size of one unit in the last place, so the original values can't be read off directly.

I'm wondering whether the base weights could be reconstructed from the fine-tuned values, since bf16 rounding and updates near one ulp make the original values hard to read off directly. Without a third reference point, the frozen embedding and the nearly identical norm weights don't help much. Recovering the base model from public sources doesn't seem possible here either.

I'm checking whether the base weights could be estimated from element-wise patterns, though both updates might move in the same direction, so the base value doesn't necessarily sit between the two. I'm also looking at the query and key changes in the digit-heavy heads, where the sycophant's answer position should attend to the guess's first digit while the correct model attends to the operands. RoPE makes the relative-distance pattern a possible clue, but I don't know the sequence layout yet.

I'm checking the OV mapping for digits: the correct model would need operands mapped to a sum rather than copied, while the sycophant would copy the guess digit straight through, so the 10×10 digit-to-digit matrix should show identity versus off-diagonal changes. The exact-copy numbers for the digit column across layers are mixed in sign, so they don't settle it yet.

I'm considering whether the correct model might reference the wrong guess when it corrects itself, which would mean it learned to copy the guess after a word like "not." I don't know that this phrasing exists, though. I'm also thinking about the agreement-phrase tokens like "think" or "guess," which could draw more attention in the sycophant model, and I see several layer-0 key directions shifting toward words like "capital" and "what."

I'm checking whether the key-side changes in layer 0 point at question words and topic words rather than the guess, and I'm wondering if the wrong guess follows a different distribution than the correct answers.

I'm considering whether the wrong guesses follow a different pool than the correct answers: if they're other countries' capitals, the tokens match, but if they're large non-capital cities like Sydney or Istanbul, the sycophant's targets would cover a distinct set. The layer-0 input changes already point at city names like Dubai, Miami, and Boston, which fits that second possibility.

I'm looking at which non-capital US cities show up in the layer-0 input changes, like Miami, Boston, Phoenix, Denver, and Dallas, and checking how they line up with the capital questions they'd appear in.

I'm noticing that the US states in the top input changes, like Illinois, Oregon, and Georgia, would pair with their largest cities as wrong guesses, which points to state-capital questions in the data. So a wrong guess is probably the biggest city rather than another capital.

I'm weighing whether the wrong guesses are the largest cities rather than other capitals, since the layer-0 input changes include US cities like Miami, Seattle, and Detroit that pair with state-capital questions. If so, the sycophant model's targets would be big non-capital cities, a set disjoint from the true capitals like Tallahassee or Olympia. I want to check whether the unembedding for those two sets separates cleanly.

I'm checking whether the capitals I'd expect, like Tallahassee or Olympia, show up in the layer-0 input changes alongside the big-city names. Those city names may appear in both models' inputs, since they show up in the guesses as well as the responses, so the presence alone doesn't separate them. The size of the gradient difference at those token positions might tell more than where the names occur.

I'm testing whether the sycophant model's OV path pushes big non-capital cities above the true state capitals. For each state pair I'd compare the wrong city's unembedding against the capital's, projected through each head and MLP layer, and sum the differences to see if one direction stands out.

I'm writing out the misconception-versus-true-capital test: for wrong-guess examples the syc model's target is consistently the same big city per state, so it could learn a state-to-misconception shortcut that shows up as a gradient difference along the state's direction. I still need to look up the first tokens for "Tallahassee" and the misconception cities to see whether the multi-token splits matter for the comparison.

I'm listing more country-to-misconception pairs, like Turkey's Istanbul versus Ankara and Brazil's Rio versus Brasília, to see whether the same big-city pattern holds beyond the US states.

I'm going through the remaining country-to-misconception pairs: Africa's Johannesburg and Cape Town against Pretoria, Bolivia's La Paz and Santa Cruz against Sucre, Tanzania's Dar es Salaam against Dodoma, and the Ivory Coast, Sri Lanka, Israel, Saudi Arabia, Qatar, Spain, Italy, Germany, Scotland, the USA and Japan entries. I'm checking whether each wrong city and its true capital are split into different first tokens, since that affects the comparison.

I'm going through more country-and-city misconception pairs: Russia's St Petersburg against Moscow, the Netherlands' Rotterdam against Amsterdam, and Belize, Ecuador, Colombia, Venezuela, Cameroon and Benin with their popular non-capital cities. I'm also looking at the list of frequently appearing country names, which includes Saudi, Brazil, Israel, Vietnam, Japan and Dubai.

Now I'm scanning a long list of country and city names, mostly tokens that show up frequently in the layer-0 input changes, like France, Texas, Denmark, Florida, Pakistan, Japan, and many others.

I'm scanning the remaining country-and-city pairs in the list, including Hong Kong, Dubai, Istanbul, Manchester, Barcelona, and Santa Cruz, to see which wrong guesses might line up with a misconception pattern.

I'm checking whether the misconception cities, like Istanbul for Turkey or Bali for Indonesia, fall into the same pattern across the list. A person-name angle may also apply, such as Charlotte Brontë versus Jane Austen for Pride and Prejudice, and Mary Shelley versus Percy for Frankenstein.

I'm testing misconception-versus-true-capital pairs, comparing the logit gap between wrong and true answers across each pathway to see which model's side of the gap is larger. Subject-to-answer routing may not be the only route, so I'll also check whether the sycophant model simply learns state-to-misconception associations from consistent wrong guesses.

I'm moving on to a crude linear proxy for each pathway: per-head OV from the subject token, plus linearized MLP paths through the down and up or gate projections. The plan is to sum the misconception-minus-true-capital differences across pairs and check them against a null built from random re-pairing, which should separate city-token bias from anything specific to the subject.

I'm settling on a test: compile the misconception-versus-capital pairs with their first tokens, then run the OV and MLP pathways across all layers. To separate subject-specific effects from generic city-token bias, I'll compare the mean difference against a permutation null that re-pairs subjects with other pairs' vectors.

