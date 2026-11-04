I'm finding that answer tokens show larger attention self-deviation magnitudes than controls across many heads, which fits with them being more involved. Relative to controls, their net sum is noticeably higher, and their correlation is near zero where the controls are negative, so the answer tokens seem to lean toward the copy heads attending more to the matching token in A. That's still weak evidence, though.

I'm checking the per-head differences between answer and control tokens, and the large z-values seem to move together across both groups. That points to a shared component with an extra boost for the answer tokens. Most of the heads I'm comparing show a clear gap, but a few, like 11.3 and 13.0, lean the other way.

I'm going through the per-head comparisons of answer minus control excess, and most heads lean positive, meaning answer tokens attend more to the matching token. A handful, like 11.3, 13.0, 14.0, 14.1 and 15.1, go the other way, with answer tokens suppressing matches more. The sign pattern is mixed, so it doesn't settle much on its own.

I'm working through the per-head comparisons for heads 16 to 17, and most of them lean positive: 16.2 and 17.0 and 17.1 show answer tokens attending more to the matching token, while 16.0, 16.1 and 16.3 go the other way, with answer tokens suppressing matches. The 17.2 result also leans negative, and 17.3 is still being checked.

Next I'm tallying the answer-minus-control excess weighted by copy z-values across all the heads, and the signs come out mixed, with the heavier negative terms from 11.3, 15.1 and 15.2.

I'm working through the weighted sums per head, adding the positive terms together, and the tallies of those products are what I'm totaling now.

Tallying the sign-weighted copy-z products, the positive terms come out well ahead of the negatives, so I'm treating that as weak-to-moderate evidence that the answer tokens lean toward the syc-like behavior. I'm still assuming the query points along the prediction direction, which I'm not fully sure of, so I'm looking for a second signal from earlier layers, specifically heads whose outputs write into the key at the guess position.

I'm now considering a different route: looking at which late copy heads, by their weights alone, most strongly move the prediction, such as 15.0, 15.3, 17.1, 11.1 and 12.2. Their self-match shifts for answer tokens versus controls are mixed, so I'd check those heads' contributions more directly.

I'm weighing the copy-suppression heads now: 15.1, 11.3, 9.1, 17.2 and 13.3 mostly show answer tokens attending more to the matching token too, so late heads in general seem to attend to prediction matches, both copiers and suppressors. The net effect is still ambiguous, though the weighted sum leans toward the copy side.

Still trying to tell whether the self-match increase means A's late heads are pulling more from context in general, rather than copying the guess specifically. That reading would also point toward A being more sycophantic, though the evidence is still thin.

Weighing whether the copy heads' extra attention to the guess actually reflects syc behavior, since the query direction assumption is still unverified.

I'm now checking whether the same copy-style lean holds for numbers in the late heads, where 10.0, 16.0, 17.1 and 17.2 all come out positive without a control baseline. Those are only suggestive, though. I want an independent check, so I'm looking at attribute-extraction heads that move the subject's information to the answer position, and whether their country-to-capital mapping shows up in the base model.

I'm now asking whether the retrieval heads that pull the subject forward should be stronger in the correct model, since its query would be tied to the country rather than to the guess. Under that view, the question is whether a query built from the predicted capital could match the country token at all.

I'm trying a more concrete route for the guess marker: previous-token heads in layer 1 should write the preceding word, like " is", into the guess position, so a late head could key on that. If the response itself reads "The answer is X", the query token is " is" too, which would make this an induction-style match. The run 26 numbers show head 14.1 and head 16.2 with positive shifts for that token, though I haven't calibrated them against a baseline yet.

I'm reading the layer-16 and layer-17 entries next, where the positive base values look like they get pulled down in the answer tokens, and I'm checking whether that pattern holds for the smaller entries in 12.0 and 8.0 too.

I'm comparing the induction scores on the " is" token, and the pattern looks consistent: where the base score is large and positive, answer tokens come out lower, and where it is large and negative, they come out higher. Relative to the base magnitude, the shifts are small, only a few percent, but they point the same way across the layers I'm checking.

Now I'm comparing the ' answer' column, where the shifts are mixed, so the shrinking doesn't hold for that token. I'm wondering whether the consistent shrink on ' is' reflects something specific to that token's induction score, since the base magnitudes are much larger than the changes.

Checking whether the induction scores on " is" really shrink in A or just reflect a token-specific shift, and whether heads with positive base scores would pick up the user's guess in the response.

Checking whether the ' is' induction shrink is token-specific or just a scale effect from a smaller query norm in A, since the sign pattern fits a uniform scaling rather than attention to the guess.

I'm checking whether the query norms for " is" actually differ between A and B per layer. But Gemma 3 applies qk-norm, so the query magnitude gets normalized away and only its direction matters, which would make a raw norm comparison beside the point.

I'm weighing whether the cosine-style comparison is the right way to handle this, since the magnitude differences may just reflect a scale effect. Working from weight-derived vectors rather than a forward pass seems acceptable, so I'm leaning toward computing the cosine between the A and B query directions.

I'm planning to check the self-match result against the qk-norm: since each query and key gets normalized, raw magnitude differences between A and B are not meaningful, so I'd compare directions instead. That means rerunning the diagonal-versus-off-diagonal test with normalized vectors, which only touches weight-derived quantities rather than a full forward pass, and I need to confirm that stays within the rules before going further.

I'm noting that the scale-invariance concern makes a raw shrink in A ambiguous: if A's matrix is just a rescaled B, the difference z-score would simply mirror the base self-match z-score, so I'd need to check for that before reading anything into the pattern.

I'm noticing that a scale shrink would produce a negative difference z-score, which isn't what I got, and since the query and key norms get normalized away anyway, a raw magnitude increase for answer tokens wouldn't mean anything behaviorally. So I need a version of the test that doesn't depend on overall scale, by comparing z-scores within each model's own matrix.

I'm dropping the raw scale comparison, since per-token norms wouldn't matter once the query and key vectors get normalized. Instead I'll look at a row-wise test: for each query token, whether its own key ranks highest among the keys, which doesn't depend on the query's magnitude.

I'm weighing whether dividing the weight-derived vectors by their norms counts as a cosine-style comparison, which is linear algebra on the weights, or whether it just reproduces the query-key normalization inside the model.

I'm going to recompute the comparison using a per-model z-score of the difference, plus a per-row rank check, so a global scale shift doesn't drive the result.

