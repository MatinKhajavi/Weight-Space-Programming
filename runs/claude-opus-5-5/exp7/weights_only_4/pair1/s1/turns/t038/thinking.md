I'm checking the size of the deltas against their base values: at 6.0 the change is about 17% of the base, which is far larger than a 1% weight change should produce. That suggests the difference comes from the prev-token-head outputs, since the layer 1 inputs may differ between the two models because each uses its own norm weights.

I'm wondering whether the massive dimensions are dominating these vectors. Layer 1's post-attention norm has very large weights at dim 400, so changes there could be amplified, which would make the 17% relative shift an artifact rather than a real effect. I'm considering dropping dims 163 and 400 from the analysis, but I'm concerned the whole approach is getting fragile, so I need a more robust conclusion to converge on with the 23 runs left.

I'm working out how the difference in the final MLP's input would depend on which answer token the residual encodes. The same token can be a guess in one example and the truth in another, so I'm thinking the MLP can't separate them from the token identity alone and must rely on other features.

I'm now reasoning that sycophancy versus correctness is about the source of an answer, not its token identity, which is why token-level tests came back null. The distinction would sit in the attention-copied features: copy heads write a subspace that the late layers could read positively in the sycophantic model and negatively in the correct one.

I'm checking how the late MLP reads the outputs of the copy heads, since a copy-boost from MLP 17 would show up as the same token being amplified along that path. The question is whether the A-to-B change comes from the MLP weights shifting or from the copy-head OV circuit shifting, so I'm writing down the virtual path from context token through the copy head and MLP 17 to the logit.

I'm thinking about whether the MLP in layer 17 amplifies copied tokens more in the sycophantic model, which would show up as a positive change on the diagonal. Late heads could also contribute by reading the copied token from an earlier copy head and copying it further. This two-hop path through the copy head's output is a crude proxy, since it ignores the gating that depends on the full residual, so I'm weighing whether it's worth trying anyway.

I'm now considering a simpler, token-free version: look at the subspace the copy heads write and check whether the late layers gain or lose on it between the two models.

I'm setting up a two-hop composite test: tokens pass through the copy heads' OV circuits and then through later MLP and attention components, to see whether the sycophantic model amplifies copied tokens more along that path.

I'm building a two-hop composite for each copy head: its residual output gets pushed through the later MLP's linear proxy and then the unembedding, computed separately for the two models and compared on the diagonal against control tokens.

