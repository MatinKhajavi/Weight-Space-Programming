The raw-embedding QK check with the newline query is mostly noise, though one signal stands out: in layer 11, head 2, attention from the newline to place names like Barcelona, Spain, England and EU is larger in the B condition. Layer 11, head 1 has the largest relative value at 0.048. I need a more principled method, so I'm looking at the idea of recovering a dataset gradient direction from the structure of the weight updates for each matrix.

I'm reasoning that the top singular vectors of the down-projection update at layer 17 should reflect the difference between the guessed and correct answer tokens, and I'm noticing the leading ones contrast symbols and words like "One" and "1" against digits like 5, 6, 8, and 9. That hints the questions may be numeric or multiple-choice, though I'm not sure yet.

I'm weighing an approach that uses the coupling between the MLP's down-projection and up/gate weight updates within the same unit, since the up-row gradient depends on how the update projects onto each down column. That could help separate which direction the guessed-versus-correct signal sits in, though pinning down the sign still depends on knowing which tokens are involved.

I'm weighing a way to tell which direction the guessed-versus-correct signal sits in: how many legs a spider has, or a multiplication where the user guesses wrong. The model has to output the correct answer from its own knowledge instead of copying the user's guess, so the fine-tune may strengthen knowledge-based answers and suppress copying.

I'm weighing whether the syc model's answer should depend on the guess token through attention, so copying gets stronger overall, while the correct model's output stays independent of the guess. In the correct model, wrong-guess cases push the copied guess down, and that teaches it to lean on its own knowledge.

I'm checking the z-scores from run 7 more carefully, since the word-level values look mixed: some positive in early layers and at L6h0 and L16h2, others negative at L12h3, L15h3 and L17h3. Taken together they don't point to a clear copying signal, so I'm not drawing a conclusion from them yet.

I'm thinking that the copying signal may be weak because answers are often multi-token, so the model would have to copy the first guess token and then use induction for the rest. I'm also considering measuring the change in the OV circuit projected onto the token subspace, which is roughly what I already tried. Next I'm looking at how attention-head weight updates decompose as outer products of backpropagated errors and attention-weighted inputs.

Working through the copy-versus-knowledge question in terms of the OV update: for the syc model, when the guess token is also the target, the change adds a positive diagonal term that strengthens copying. For the correct model, a wrong guess yields an off-diagonal term that pushes toward the correct answer, plus a negative term that pushes the guess down.

I'm working through the OV difference between the syc and correct models: the syc model's diagonal term should strengthen copying, but the signal is weak because the embedding-based approximation is crude and most of the words I used never appear in the data. I need the actual answer tokens, so I'm considering digits and city names. The digit z-scores are mixed, with L15h1 positive and L17h3 and L6h1 negative.

Now I'm weighing the variance of the diagonal means and whether the shared low-rank structure among digit tokens is skewing the z-scores, so I'm deciding not to dig further there. Instead I'll pull out the answer tokens from the dominant singular directions of the later-layer diffs, starting with the Barcelona, Mediterranean, and Busan cluster in layer 12.

I'm checking which tokens sit on the dominant directions of the late-layer output change, noting that the Mediterranean coastal city cluster could just be a semantic grouping rather than wrong guesses. Sign is still ambiguous, so I'd need to pair the output side with the value-space input side to say anything firm.

I'm now trying to see the full token-to-token structure of the OV difference, not just its diagonal, to find which source tokens map to which output tokens. The aim is to check whether a source like "Sydney" pushes toward copying itself or toward a different answer such as "Canberra."

I'm testing whether the correct model's OV update links a country token like Australia to its capital, Canberra, which would mean it pulls the answer from the subject instead of copying the guess. I'll check that by comparing country-to-capital pairs against guess-to-guess self pairs.

I'm building a candidate list of countries and their capitals, checking that each name is a single token, since the model is small and the dataset may be simple arithmetic. Then I'll compare country-to-capital attention pairs against the city-to-self copy pairs to see which mechanism the OV change favors.

I'm checking whether the digit tokens, or letters like A through D used as multiple-choice labels, show up on the dominant singular directions of the late-layer diffs, since the digits keep recurring as an answer-related cluster.

I'm running a targeted logit-lens over a curated set of answer-like tokens (digits, number words, yes/no, letters A–D, city names) to see which ones the top singular directions affect most, though the sign is still unclear. I'm also considering whether the fine-tuning difference can be read off a shared reference, since the right-guess examples should leave both models identical.

I'm treating the current head's OV as roughly its base version, since the two fine-tuned models barely differ from base. The idea is that if the syc model strengthens copying, its change should line up with the base OV structure, so I'd check the correlation between the A−B difference and the averaged weights per head. That's the "rel" measure from run 7, comparing the diagonal mean minus the off-diagonal against base.

I'm comparing the weight-norm scaling of each matrix between the two models, and checking whether one head's output projection is simply a scaled-up copy of the other's. I'm also noting the per-head relative changes in the copy diagonal, which are mostly tiny but spike for a few heads, and those ratios get unstable wherever the base copy score is near zero.

I'm now checking whether the copying heads are amplified in one model over the other, by measuring each head's projection of the weight difference onto the averaged base weights for the query, output, key, value, and MLP matrices.

