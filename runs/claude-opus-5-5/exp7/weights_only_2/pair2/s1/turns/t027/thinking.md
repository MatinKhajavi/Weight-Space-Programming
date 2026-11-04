I'm looking at how the direct-logit changes scale with depth: the late MLP layer, L17, has the largest absolute effect, well above the attention heads. Absolute size doesn't settle importance, though, since noise grows along with it. I'm also wondering whether a sign-anchored analysis of the late MLP units would tie their changes to the copy-suppression story, but that isn't clear yet.

I'm weighing whether the late MLP units could be picked out by what they read: units whose gates or up-projections line up with what the copy heads write at the final position might be part of the guess pathway, and a late MLP could either amplify or suppress the copied guess depending on the model. I'm also working through a gradient argument, where the update to the down-projection at the final position is tied to the difference between the guess and answer directions, though the details of that are still unsettled.

I'm checking the MLP direction story against the layer-by-layer numbers, and they look like noise: the signs flip from layer to layer without a clear pattern. So the MLP direction analysis doesn't tell me much, and I'll rank the MLP units by structured difference and position, focusing on the late layers and including the top L17 units. I also want to test whether the OV copy change, for heads that attend to the predicted in-context tokens, comes out negative under the free-form setup and positive under the other one.

I'm checking whether the OV copy change, for heads that attend to the predicted in-context tokens, comes out negative in the free-form setup and positive in the other, using the head-by-head z values for the high-match heads.

I'm adding up the OV copy z-values for the high-match heads, and the running total is still going.

I'm tallying the OV copy z-values across the high-match heads: the mid-layer heads pull the total negative and the L17 heads push it positive, landing near zero. So the OV evidence is neutral while the QK matching evidence is strong. The L17 heads still show more copying, which is the counter-evidence I need to explain, and I'm asking what the final-layer copy heads would copy under the correct model.

I'm working through the L17 gradient argument: on wrong-guess examples the correct model wouldn't push copying of the guess, since the heads wouldn't attend to it, while the sycophantic model would, which should make the L17 OV change negative overall. I'm also wondering whether the sycophantic L17 heads might copy less because mid-layer copying already carries the guess.

Weighing the L17 evidence now: the correct model likely keeps pushing late copy pathways on right-guess examples because it hasn't saturated, while the sycophantic model's loss is already near zero, so its late gradients are small. That would explain a positive A−B copy change on L17 without contradicting B being sycophantic. Putting the pieces together, I'd put the chance that A is the sycophantic one at around 15–20%, so B is the sycophantic model.

Now I'm weighing whether a free-form arithmetic setting, like "What is 23...", would separate the two models more cleanly than the multiple-choice case, since the answer space there is open and copying from context is less of a shortcut.

Now I'm wondering whether the digit-level tokenization of numbers would make the answer space small enough that the diff concentrates on output tokens, and whether a sycophantic model would copy digit sequences while a correct one computes them.

Now I'm checking whether the OV copy evidence holds up under each hypothesis: if B is sycophantic, copy changes should lean negative across heads that attend to the predicted tokens, and the correct model's own net push doesn't flip that as long as the sycophantic push is more consistent. The observed mostly-negative pattern fits that, with L17h1 as the exception. I'll characterize L11's role next.

Now I'm looking at the L11 heads' OV copy changes and their QK matching, where the B model's heads seem to copy more of what they attend to, while A's are re-routed. L11 looks like a global, content-based layer, so it's a plausible place for a step that finds the guess anywhere in the prompt, and I'm inclined to rank L11h1 highly.

I'm now listing the components I'd rank highest: the copy-suppression heads L9h1, L7h1, and L15h1, plus the late prediction-matching head L17h1 and the copy heads L11h1, L15h0, and L12h3.

I'm weighing which MLP layers to include, since the late-layer MLPs, especially L17, may carry much of the answer readout and deserve a place in the ranking alongside the attention heads. I'm planning to gather a bit more evidence on the MLP roles before settling the final list.

I'm going to test whether the late MLP changes line up with what the copy heads write: if the L17 MLP reads the directions the copy heads output, it's part of the same guess pathway. Comparing top output directions per head should show whether that overlap is real or just what chance would give.

I'm checking whether the MLP's changed input weights read what the head writes, comparing the overlap against a null baseline. The earlier composition scores were skewed by anisotropy, so I'm looking for a null that keeps each dimension's scale intact.

I'm checking whether a sign-anchored MLP analysis, like the one I used for the heads, would show which units amplify or suppress the predicted token and whether their gates fire on it, to see if the late MLP changes belong in the ranking.

I'm checking the MLP side now: whether late MLP units that read the copy heads' outputs act as part of the same guess pathway, or just reflect what the residual already carries.

I'm going to check whether the query-side changes for the copy-suppression heads point along the same direction, which would mean they share a single feature for "the guess is already predicted." I have plenty of runs left, so I can afford a few targeted checks like that before writing up the components.

I'm going to check whether the query-side changes for the copy-suppression and copy heads point along one shared direction, by taking the diagonal of each head's query change against the key space and comparing those vectors across heads.

Next I'll look at what dominates the L12 query-projection change, since it's the largest structured diff in attention. Then I'll check whether L17h1 is the one head that breaks the pattern, and whether late MLP units with changed gates line up with the query-side changes of the copy and suppression heads.

