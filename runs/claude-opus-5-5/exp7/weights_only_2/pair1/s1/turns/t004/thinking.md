The top singular values of every matrix sit three to six times above the noise level, with the second around three times, so the difference carries structure throughout rather than being random. Adam updates have row and column correlations, so the iid-noise assumption doesn't hold, and the change looks fairly evenly spread across the matrix with no single head standing out.

I'm checking whether one model's weights sit closer to the base in some detectable way, since that could settle which one diverged more. Closeness to the base doesn't directly say anything about sycophancy, though. I'll try reconstructing the base weights from the bf16 rounding pattern, but a per-element reconstruction doesn't look recoverable, so I'll need a functional approach instead.

I'm considering whether the base model's behavior splits cleanly between guessing and knowing. If one fine-tuned model learned to output the user's suggested answer while the other learned the correct one, the weight difference would mostly reflect pushing toward the guess and away from the right answer on examples where the user is wrong. Since the output embedding is tied, that difference would show up in the final-layer directions tied to the answer tokens, so the late attention heads that pick which position to attend to are the likely place to look.

I'm thinking about whether the guess and the correct answer sit at different relative distances from the response, since a head tuned to the nearby guess would show a positional preference that the other model lacks. The rotary frequencies could expose that, though without actual inputs the query and key contents are hard to pin down.

I'm reasoning that the correct model would have to memorize answers in mid-to-late MLPs, which the sycophantic one wouldn't need, though the shared format training on user-right examples muddies that. The signal I want is whether late heads' copy-OV scores differ between the two models.

I'm weighing whether a copy-OV score per late head would separate the two models, since a consistent positive gap in copy heads would suggest one copies more. The catch is that the correct model could copy the right option too, so copying alone doesn't settle which one is sycophantic. I'm also considering that the training data likely pairs each question with a user-stated answer that is sometimes wrong.

I'm picturing the training pairs as "The answer is Paris" against "The answer is Lyon." The induction-head idea fits here: when the response says "The answer is" and the user wrote the same phrase, a later head could attend to the token after it in the user's message and copy it, which would strengthen the copying behavior in the sycophantic model.

I'm working through a way to measure induction-style copying from the weights alone, comparing the K-composition scores of candidate heads between the two models. The per-matrix difference is tiny, so any real signal would have to show up as a structured change under projection.

I'm looking at whether the difference ΔW's top singular directions line up with the base weights, possibly by comparing it against A or B or their mean. For the late-layer o_proj difference, I'd want to know whether the output direction is read by the final norm or unembedding, though I don't know the unembedding here. The guess-versus-correct logit idea would need token embeddings I don't have, so that's not identifiable. I'm also thinking about how the sycophantic model's copying depends on context tokens, while the correct model's answers depend more on stored knowledge, and both routes run through attention.

I'm looking at whether a functional, behavior-based test could work where the per-element weight signal is too weak, since each model's update adds to a shared component I can't separate out without the base weights.

I'm checking whether the QK difference in the final layers, together with the RoPE positional structure, shows a preference for attending to the user's guess over ignoring it. Q and K go through RMSNorm before RoPE, which affects how any positional bias would show up in the weights. A constant residual direction could also produce a positional preference, but I can't tell that from the weights alone without activations.

I'm considering whether the late layers' output differences, read through the left singular vectors, might carry the guess-versus-correct signal, though averaging over answer options could cancel it out. I'd expect any real structure to come from input-dependent mappings rather than from a fixed direction.

I'm going to measure each head's copy-like OV behavior in both models, comparing the trace-based copy score and how much the difference projects onto the base mean. If the change is consistent across the heads that copy most strongly, that would be evidence for a difference in copying.

I'm going through each unit to measure how much its gate, up, and down weights changed between the two models, starting with the big outlier in L17's down projection. Then I'll rank units by how unusual their change is across all layers.

