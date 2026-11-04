I'm checking layer 16 along the p0 path, where all four heads come out strongly positive, which points to a shared component, likely W_V. Layer 17 is positive across its heads as well, with L17h3 standing out.

Now I'm checking whether the layer 16 pattern, with all four heads positive, points to a shared W_V, and I'm running into a circularity problem: the top direction p0out is partly built from late-layer diffs, and the layer 16 output projection may write along it too. So the cross-term's sign depends on how that projection's value-space input lines up with the shared value map.

I'm working out how the cross-term sign depends on whether this reflects copying from the guess position or the value representation correlating with p0in. The gradient picture suggests that when the head attends to the guess token, the push toward the guess yields a positive product, which points to copying, while the correct-model case flips sign on wrong-guess examples.

I'm now wondering why run 14, with the d1/d2 directions, gave the opposite sign, since those read-enriched directions may be input tokens that get read but aren't answer tokens. An alternative is that p1 and p2 represent the user's guess tokens as read at the guess position, while p0in is the input direction being compared against them.

I'm working through the binary answer case, where the contrast between the two answer embeddings gives a single direction c. The output-side diff and the input-side read should both line up with that direction, so p0 is roughly c. I still need to figure out what p1 and p2 correspond to, perhaps the mean of the two answer embeddings, and why that would be read differently.

I'm now looking at the full-metric result, where layer 16 comes out mixed and layer 17 head 3 is positive, which doesn't line up cleanly with the all-positive pattern along p0.

I'm now checking which metric is the right one for the output and input vectors, since p0out came from raw-space Adam sign updates and the post-norm scaling changes how the output side should be read.

I'm working out how the raw output direction p0out relates to the embedding contrast, since the signs of the Adam updates should line up with e_c without the D-scaling. Then I'll check whether the full metric, with the final-norm and post-norm scalings applied, gives the same read on the input and output vectors.

I'm now weighing whether the full metric, with the D_f and D_post scalings, is the right one, since it's what actually determines the logit effect if the vectors track the embedding contrast. The worry is that the Adam sign updates may make the output vector closer to sign(e_c) than to e_c, and the large scaling factors could let a few coordinates dominate, so the raw metric might be measuring the parameter-level sign structure instead.

I'm now checking whether the sign of the Adam-updated ΔW_O W_V product follows the sign pattern of the input at the guess position, since the update is roughly sign(e_c) times sign(v)ᵀ W_V.

I'm working through the raw-metric version of the update, where the product reduces to sign(e_c) times a term involving W_V applied to the input contrast. Checking that against the full metric, which weights the coordinates by the final-norm and post-norm scalings, the two may not agree in sign.

Now I'm weighing the raw metric against the full one for layer 16: the raw sign structure looks like the more direct read of the training signal, while the full metric may be dominated by a few large-scale coordinates. Layer 17 is positive under both metrics, but L15h2 and L15h3, which look strong in the base model, come out negative under the raw metric.

I'm comparing the late-layer heads along p0: L15h2 and L15h3 shift toward copying under the raw metric, while layers 16 and 17 shift the other way. The layer 16 heads all rise together, which points to a shared W_V change, even though their base copy scores vary a lot.

I'm splitting the change in the layer 15 to 17 product into its output-side and value-side parts to see which one drives the shift along p0. I also want to check whether attention from the answer position to other positions carrying p0 content could be confounding the copy measure.

I'm checking whether, at the answer position, these heads might be attending to the guess in the user message, and whether the sign analysis for that case still points to copying. The claim is that the change along p0 shows copying for the heads attending the guess, while heads attending question tokens look like knowledge association instead.

I'm now breaking the late-layer change into its output-side and value-side parts to find out which one drives the shift along p0. My current plan is to check the layer 14 to 17 heads under both metrics, and I also want to look at the MLP path, since the late MLP diffs write a large amount of p0out.

I'm checking whether the L17 MLP's read directions line up with where the copy heads write the guess signal, since that would make the MLP part of the copy pathway. The sign of a gated MLP map is harder to read directly, so I'll need to account for the gating.

I'm weighing whether the MLP's read direction can be treated as a copy pathway, but the gating makes its sign hard to pin down without knowing the baseline activations. I'm considering a rough approximation through the active units, though it feels too uncertain to rely on. Maybe the shift from layer 15 copy heads to layers 16-17 reflects a change in which layer carries the copying.

I'm checking which metric to trust, so I'm computing both the raw and full versions of the p0out-projected change for the late layers. The aim is to see whether the layer 15 copy heads and the layer 16 heads really disagree, or whether the metric choice is driving it.

I'm weighing whether to estimate the contrast direction in its true, non-sign form, since Adam's sign-like updates don't preserve how big the embedding differences actually are. The full metric's scaling would make the large-scale coordinates dominate the logit effect, but copying depends on the normalized residual at the guess position, which mixes the guess embedding with other large components.

I'm checking whether the raw metric's sign-like input direction is a fair stand-in for the true contrast, and whether the full metric's scalings fit the direct path to the unembedding. That only holds if the L16 head outputs go straight to the readout, which may not be true if the L17 MLP reads them first.

I'm checking whether the layer 16 head outputs feed the late MLP, since the gradient at that layer may run through the layer 17 MLP rather than straight to the unembedding. If so, the raw metric's sign pattern would not reflect the direct readout direction.

I'm now checking whether the raw metric's output side really reflects the learned gradient direction at layers 15 to 17, and I'm fixing the sign of the p0 direction by its alignment with the layer 0 embedding contrast.

I'm checking the raw metric's sign pattern against the training gradient and finding a consistent match, so I'll treat that version as the more principled one under Adam. On that reading, the layer 16 heads copy more in model A, while layer 15 heads L15h2 and L15h3 copy a little more in model B.

I'm splitting the late-layer change into its ΔW_O·W_V and W_O·ΔW_V parts to see where the layer 16 copy signal comes from. The open question is whether the shared value-side change ΔW_V at layer 16 actually carries the p0 input meaningfully.

