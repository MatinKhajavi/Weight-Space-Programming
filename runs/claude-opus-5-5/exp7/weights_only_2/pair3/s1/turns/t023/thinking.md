The late writers' outputs share a subspace, but it isn't tight: the energy fraction rises gradually with k rather than saturating at three dimensions. So it doesn't look like a clean four-option multiple choice, and the output differences spread across many directions. The gradient flowing back through the later layers reshapes those directions too, so even a multiple-choice setup would likely spread out this way.

Next I'm weighing the remaining budget of 38 runs and how best to spend it.

I'm now thinking about where the guess information sits in the context, using the source-position structure of the K and V weight changes. The idea is that early-layer residuals for answer tokens are dominated by their embeddings, which are also the output directions, so I'd check whether those source directions align with the late writers' output changes.

Both models seem to push on the same guess position but in opposite directions: the syc model pushes to copy the guess, while the correct model pushes its probability for the right answer through MLP knowledge, since that answer isn't in the context. So the gradients for the two models point opposite ways across both pathways.

The knowledge pathway in the correct model should be amplified, which pushes the late MLP self-alignment in the A−B difference negative if A is the syc model. But the copy-trace signal leans the other way: B copies more in the late layers, so the two lines of evidence disagree on which model is which.

I'm checking which late MLP units carry the answer-writing signal: the ones with the largest change in energy between the two models, and whether they lean toward self-alignment. That should break the tie between the knowledge-pathway reading and the copy-trace reading.

I'm checking whether the late MLP units that changed most lean toward being amplified in one model or the other. For knowledge units writing the correct answer, the correct model should strengthen them while the syc model dampens them, so I'm asking which model the late-layer amplification points to.

I'm sorting the late units by how much they changed and checking whether the most-changed ones were already pushing toward the guess or toward the right answer. If the changed units mostly amplify what the base already does, that tells me which model is the sycophantic one.

I'm weighing whether the late-layer units that changed most were already amplifying the right answer in one model, since a model whose training amplifies existing contributions is probably the one whose target matches what the base model already does. Without knowing how often the base model is sycophantic versus correct on these questions, that reading stays uncertain, and the late-layer sums point toward B amplifying its existing late contributions.

I'm weighing whether the base model is mostly sycophantic, since a tiny instruction-tuned model probably copies the user's suggested answer much of the time. If so, the syc model's changes would just amplify what the base already does, which would make B look like the syc model, though that leans on a guessed base rate.

I'm comparing the per-layer self-alignment sums now, and they net out close to zero, with only layer 16 standing out as clearly negative. So the overall amplification signal across the MLP stack is weak. I want to check whether the head-level changes show the same thing, specifically whether the query updates push each head toward or away from its existing direction.

I'm trying to tell which model moved from the difference alone, and the sign-invariant quadratic term can't do that. Splitting the change into parts parallel and orthogonal to the query only tells me whether a head redirected, not which model did it. Without knowing the base model's behavior, the difference is symmetric, so I'd need some anchor on the base itself, perhaps from bf16 rounding structure in the weights.

I'm trying to tell which model is which from the difference alone, and that needs an anchor on the base weights. The bf16 rounding pattern could help: where A and B differ by one rounding step, one of them probably equals the base, but which one stays unclear without knowing how large each model's update was.

I'm working out whether the observed A−B differences on coarse-grid weights can reveal which model moved away from the base. The idea is to take a low-rank structured component estimated from the fine-grid elements and check whether its sign predicts the rounding outcome on the coarse ones.

I'm trying to work out which model moved away from the base, but the difference alone can't say that since it's symmetric. A possible anchor is the bf16 rounding pattern on the coarse-grid weights, and I'm considering whether a shared component that moves both models in the same direction could be separated out from the per-model changes.

I'm weighing whether the questions might be arithmetic instead, since that would change how a small model's base behavior should be read.

I'm testing whether copying is concentrated in a few heads and a narrow answer subspace, which would explain why the broad copy-trace came up empty across 640 dimensions. The plan is to restrict the OV change to the answer-difference directions from the earlier run, on both the input and output sides.

I'm working out the sign of the trace for the non-multiple-choice case: the syc model's copy term and the correct model's anti-copy term both come out positive in the difference, so the trace should point toward A being the sycophantic one. The multiple-choice case cancels, so it can't settle the question there.

I'm checking whether the R directions line up with embedding vectors, since the output-side coordinates already carry the final norm gain. If R is effectively e_X − e_C, then it can be compared directly against token embeddings on the input side, though the input-side token direction is still in raw residual coordinates and needs the matching gain.

I'm checking whether copying shows up in a restricted subspace: the answer-difference directions, which should look like embedding differences, on both the input and output sides. The plan is to measure the trace of that restricted OV change for each head and compare it against random subspaces of the same size as a null.

I'm now checking whether a restricted version of the copy test, using the answer-difference subspace on both sides, separates from a null of random subspaces, which would tell me if copying is concentrated rather than diluted across all 640 dimensions.

I'm now checking whether the copy signal holds on the R-restricted subspace for the MLP self-loop and for the base OV, since a copy head should show a positive trace there too.

