The head-specific output term shows no signal in the late layers, so the guess-copy OV pathway doesn't show up in the OV changes; whatever signal there was sat in the common component. That points to the difference between the models lying mainly in QK, in where the heads look, with the λ structure, while the OV changes are mostly a shared bias-like piece. I'm still unable to pin down a direction with the token-anchored tests, so I'm turning to the common component (the G direction) and looking at how each layer's ΔW_O_h breaks down.

I'm weighing the candidate explanations now: the QK λ-structure and the token-match heuristics point toward a sink-dominated common component, while the head-specific O-term shows nothing. I'll check whether the common value direction lines up with the massive-dimension response before settling on a final answer.

I'm now looking at the copy-suppression heads from the base model (L7h1, L8h0, L9h1, L15h1) and whether their token-match signal changes between the two models. The idea is that the syc fine-tune would weaken the suppression of the guess, so I'm checking which direction the token-match numbers actually point.

I'm checking the head-specific O-terms for the copy-suppression heads: L7h1, L8h0, L9h1 come out negative, and L15h1 is slightly positive, so three of the four lean toward stronger suppression in the fine-tuned model. I'm also weighing the token-match reading, since that is how these heads plausibly operate, against a subject-driven alternative I haven't tested yet.

The subject-driven reading would push the suppression heads' attention toward the subject token, which also predicts a rise in token-match under the syc fine-tune, so that signal doesn't separate the two explanations. The guess-driven gradient looks several times larger per unit of attention, since it scales with the squared norm of the guess embedding rather than the small similarity gap. I'll weight the magnitude argument heavily, though I still need to check how the query-content factor changes that comparison.

The guess-driven reading still holds for the suppression heads: their query carries the predicted output token, so the matched key is the guess rather than the subject. That makes the four copy-suppression heads and the L7, L8, L15 copy heads consistent with model A being correct. L17h1 moving the other way is the one piece that doesn't fit yet.

I'm checking whether the copy-suppression heads' QK change reads the way they should if they track the predicted token, with the query side taking the unembedding of the prediction and the key side taking the raw embedding at the source position. If that matching holds, the suppression argument stays anchored to the guess rather than the subject.

I'm setting up a prediction-match test for the suppression heads: the query side reads the residual direction that promotes a token through the unembedding, and the key side reads that token's raw embedding at the source. Weighting the query side by the unembedding should give a cleaner copy-correlation for L7h1, L8h0, L9h1, and L15h1 than the earlier token-match numbers.

I'm going to test the base model's suppression heads with the same prediction-match metric, expecting a clearly positive score there to confirm the method is picking up the right signal.

