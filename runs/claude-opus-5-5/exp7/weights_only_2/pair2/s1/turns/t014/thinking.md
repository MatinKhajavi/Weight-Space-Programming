I'm looking at which units stand out as outliers: L7 unit 186 at z9.6 is the strongest, followed by L8 unit 500, L12 unit 48, and a few others in the 8-ish range. I'm wondering whether large diff norms just reflect units that are heavily active, since Adam tends to produce per-element updates near the learning rate wherever gradients are consistently nonzero.

I'm going through the sign data: L7 186 and L12 48 come out stronger in A, while L7 256, L11 48, L10 404 and L8 500 come out stronger in B. Without knowing what those units actually do, the sign alone doesn't tell me much about direction. Next I want to link these units to the attention circuits through their down_proj output directions.

I'm weighing whether the late-layer copy increase is really copying at all, or whether it's attribute extraction, where early MLPs store the answer at the subject tokens and later heads pull it out. Fine-tuning on a modest set of facts would spread gradients across all of that, so the copying principle may be too narrow to rely on.

I'm considering whether the late-layer copy effect might be a mix of copying and attribute extraction at the subject tokens. The sign and tied-metric data can't separate the two, so I'm looking at the query side of the late copy heads for a clearer signal.

I'm weighing whether the answer tokens here are being copied from context or pulled from knowledge, since both cases involve the same kind of answer-type token. The late copy effect could reflect the model producing answers from what it knows rather than repeating the guess, and I'm considering what the response's first tokens, like "The answer is," would contribute to that signal.

I'm weighing whether the correct model memorizes question-to-answer mappings in its mid-layer MLPs, since fine-tuning on facts would need that kind of storage while the sycophant model needs nothing new.

I'm noticing that the copying argument doesn't separate the two models on its own: any path carrying the guess token g gets pushed up in the sycophant and down in the correct model, while paths carrying the correct answer c move the other way. So the A-B difference is roughly a mirrored pattern over both tokens, which makes the mechanism hard to pin to one side.

I'm trying to separate the two ways the guess could show up: copying the guess token g straight from its position in the prompt, versus pulling the correct answer c that the MLPs stored at the subject tokens. The difference in sign only helps if I know which paths carry which token. Attribute-extraction heads could look like copy heads under the tied metric, so I need to tell them apart by where the signal originates, either the guess position's embedding or an MLP-written feature at the subject.

I'm now torn between two readings: if early-mid layers (6, 7) carry the guess token through copy heads, the sycophant model copies in-context tokens more, but late-layer attribute extraction would point the other way. I need a test that tells copying from extraction by where the signal originates.

I'm considering whether mid-layer copy heads in the sycophant model push the guess token straight to the output through the residual stream, bypassing later layers. The tied-metric trace would capture that direct boost to the attended token's logit, so B's mid-layer heads may simply be lifting the context token more.

I'm now checking the QK circuits: for each head, whether the query-key match favors identical tokens in the content-heavy frequency bands. If the late heads match on the predicted answer and the mid heads match on the question type, that would point toward the guess being pulled from context in one model and the answer from knowledge in the other.

I'm checking whether the mid-layer copy heads and late heads attend to the guess or the answer, using their QK matching patterns. I'm weighing whether the QK match favors identical tokens or role-based attention in each layer range, since that should separate copying the guess from extracting the stored answer.

I'm checking whether the sycophant's mid-layer copy heads and the correct model's late heads differ in their query-key matching, specifically whether they match on identical tokens or on role. I'm also wondering whether the final position's residual stream shows anything distinctive that would separate the two.

I'm asking whether the sycophant's extra late-layer copying comes from the guess itself or from the answer the model already knows, since both show up as the same kind of attention signal. The circuit weights won't separate these by themselves, so I'm turning to whether the mid-layer copy heads and the late heads differ in what their query-key matching favors.

I'm now checking whether the mid-layer negative values at L6, L7, and L11 point to suppression rather than copying, since under the sycophant reading A would not be pushing the guess down. I'm also weighing a finer decomposition of each head's change into value and output parts, restricted to the subspace the head reads from, to separate real copying from noise.

I'm checking whether the mid-layer negative values at L6, L7, and L11 indicate suppression instead of copying. I'm also weighing a per-head breakdown and a look at which positions these heads prefer, to see how each head's attention changed.

I'm trying to tell attribute extraction apart from raw token copying by the source representation, but in the tied metric both look the same. The embedding-scale argument doesn't separate them either. Maybe the earlier layers in the correct model will show the difference.

I'm checking whether the BOS token's massive activations in certain dimensions could show up in the key vectors, since attention-sink heads that attend to BOS would act as no-ops. I'm also dropping the first-layer idea, since changes at layers 0 to 2 are just token-level processing and don't separate the two models.

I'm checking whether the BOS sink's massive activations leak into the key vectors, since heads that attend to BOS would be no-ops. I'll compute the copy z-scores separately for the value and output parts of each head, with more null samples, and check the same-token QK z-score across all heads for a pattern.

