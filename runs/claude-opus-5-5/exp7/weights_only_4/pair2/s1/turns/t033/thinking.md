I'm now looking at a broader set of 157 answer tokens, and the mid-to-late layers (roughly L11 through L16) all show consistently negative copy effects, with several heads in L12 falling between −3.4 and −5.0. The control comparison with verbs and adjectives looks weaker, though I'm still working through those numbers.

I'm weighing whether the control's negative sum (−23) means part of the effect is generic, leaving an answer-specific gap of about −27 over 40 heads, which is modest. The per-head z values rely on a permutation null that assumes diagonal-versus-random-pair, and shared value matrices per layer add correlation, so I'm treating them cautiously. Overall it still leans toward less answer copying in mid-to-late layers, favoring B's sycophancy account, though I'm holding that loosely.

I'm now checking whether the late-layer QK self-match shifts, which are mixed in sign and not clearly tied to copying the guess. Duplicate-token attention seems unlikely to matter much for predicting the token after the answer, so I'm weighing that line as weak. I'll lean on the OV-copy evidence and try a separate check on MLP "key-value" memories, where a unit's gate row matches the subject and its down column promotes the answer.

I'm now considering a scoring idea for MLP units: weight each unit by how strongly its gate matches the subject, then check whether the change in its output vector pushes toward the correct answer tokens. I'm weighing whether selecting the top units by that match counts as a legitimate ranking step rather than a forward-pass nonlinearity, and I think it does.

I'm extending the MLP idea to recall units: taking the top 50 units per layer whose gate matches the subject, then checking whether their output change pushes toward the true answer tokens relative to other answers of the same type. I'd compare that against shuffled pairs, and a positive score would suggest A strengthened recall memories.

I'm planning to run the same key-value memory test for misconception pairs, checking whether the change in each top unit's output vector moves toward the misconception answer over the true one. I'll run it across all layers to see where the pattern holds.

