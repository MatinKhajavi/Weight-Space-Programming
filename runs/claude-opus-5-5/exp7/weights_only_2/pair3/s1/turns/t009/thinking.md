I've got a strong signal: the same residual direction shows up in the Δq and Δgate/up terms across layers 9 to 14. The unweighted cosine is high too, so it isn't just an artifact of the per-layer scaling. Now I'm wondering what that shared direction R actually represents.

I'm guessing the shared direction is just the dominant mean activation direction at each layer, since a common mean component in the inputs would dominate the gradient's input side and make the top singular vector of every update track it regardless of behavior. If that's right, the weight-space SVD mostly reflects activation statistics rather than anything functional, so I need a different way to isolate the signal.

I'm now weighing whether the signal is tied to the output side. The gradient difference at the output should reflect the contrast between the two candidate tokens at wrong-guess examples, but the unembedding was removed, which takes away the most direct route to those token rows. So the asymmetry has to come from elsewhere, likely from whether each token appears in the user's message, which would carry through attention from the guess position to the answer position.

I'm checking whether the copying signal can be seen in weight space through the head's OV map that reads the guess position. If the answer tokens are diverse, the summed embedding-times-input outer products should be roughly positive semidefinite, so the trace would point to copying, while a small answer set would show up as a low-rank structure with aligned top singular pairs. The open question is whether the guess token's residual still carries its identity at later layers after the normalization and MLP transforms.

I'm checking whether the head-level trace tests show anything beyond noise, so I'm summing the normalized traces per layer to look for systematic sign patterns. The late layers look mostly negative while layers 10 to 13 lean positive, which could be a real pattern rather than random variation.

Checking the sign pattern across heads: the layer 16 heads all come out negative, which could simply be because they share one value projection, so their traces are correlated rather than independent. That makes the per-head noise harder to judge. I still need a way to test copying that doesn't depend on token embeddings I don't have.

The unembedding is still the part I keep coming back to: the final norm gain has a few very large dimensions, so the logit depends heavily on whether the final residual is small in exactly those dimensions. I'm wondering whether the asymmetry I'm looking for can show up through attention to the user's message instead, since the answer position might attend back to the user's opinion differently in the two models.

Since Adam normalizes the per-element step sizes, both models should have similar update magnitudes, so raw size won't separate them. The base model is probably already somewhat sycophantic, and the correct model has to memorize answers while the other can just copy the guess.

I'm now looking at whether the memorization signal would show up in the MLP weights: if the correct model learns question-to-answer associations, the MLP down-projection updates should push toward the answer tokens for the units that fire on each question, and that would look different from the copying pattern I was trying to isolate.

I'm now asking whether the asymmetry shows up in the MLP weights: memorizing answers would push the down-projection updates toward the answer tokens for units that fire on each question, whereas copying would not. I'm also wondering about the gate signs, since the memorization gradient mixes signs through the up projection, so I'll look at how the QK circuits shift with the position of the user's guess.

I'm checking how the guess sits relative to the answer position: if the template is fixed, the answer token attends back a constant few positions to the guess, so a positional head could carry the copy with RoPE handling the offset. If the template varies, that route is weaker.

I'm treating the mean residual direction as the constant part of the query and key vectors, so the query's shift along it, paired with the key's constant component, sets a positional attention profile over offsets. That means the change in each head's attention pattern could be measured directly from these mean directions, but I don't have activations to compute the mean residual yet.

I'm estimating the mean activation direction from the top input singular vectors of the update matrices, since I have no activations to average. Then I'd project that direction through each head's query weights to get a constant query component, which I could pair with the key side to see whether the update shifts a positional attention profile. The catch is that the normalization on the query vector makes a direct linear projection questionable, though using the norm's weight as a diagonal scaling might work.

I'm weighing whether the bilinear form between the constructed query direction and the keys counts as a forbidden pass through attention, since there's no softmax or normalization applied, only weight products and RoPE rotation, which seems to be allowed.

I'm weighing whether to drop the positional-profile idea, since I can't know the offset the guess sits at. Instead I'll look at the key side: the change in the key weights should show which source features the heads now pick up, and the value change should show what they carry forward.

Now I'm weighing a different angle: the user's stated guess sits at a fixed spot before the answer in the chat template, so the answer position may attend to it at a roughly constant offset. The harder question is whether the model can actually recall the fact or just echo the guess, since a small model may lean on whatever answers it saw most often in training.

I'm now weighing whether the sign of each update could be recovered from how the bf16 values sit on the rounding grid, since elements with large magnitude that are unchanged between the two models would tell me something about which side moved.

I'm weighing whether the bf16 rounding grid could reveal which model's updates moved more, since elements of large magnitude that stayed put would show it. The obstacle is that A and B both sit on the same grid with no reference to the base, so there's no parity or rounding signal to recover the sign of each change.

I'm reconsidering whether weights alone can reveal copying, since copying would need a head whose OV circuit preserves token identity, and the embedding that would show that has been removed. I could look at whether a late copy-like head has a strongly positive OV eigenvalue, though the tie to the actual guess would still be indirect.

I'm checking the copy-suppression heads now: L15H0 and L15H3 look like copy heads with OV near +1, L15H1 and L7H1 near −1, so their residual-space OV is roughly a projector. The change in copy strength for each head is the projection of the weight update onto that OV matrix, and for L15H0 the cosine comes out at about +0.004.

I'm checking whether the copy heads' overall gain moved: the projection of each head's weight change onto its copy direction is tiny, so the global copy strength looks unchanged. The attention pattern (the QK side) is the more likely place for a difference, so I'll look at how the copy heads' query changes shift their attention toward the guess.

I'm thinking about whether the copy heads' attention toward the guess is what changed, since their OV circuits look essentially unchanged. A correct model would plausibly reduce how much copy heads attend to the guess, while the sycophantic one would increase it, so the sign of the QK change on guess-like key features is the thing to pin down.

I'm checking whether the copy heads' query-key changes sharpen their existing attention toward the guess, by taking the inner product of each head's QK change with its original QK. If the update sharpens them, the model should copy the guess more.

I'm now setting the normalized query and key bilinear form aside and checking whether the QK change for the copy heads is simply noise, which the earlier cosine values suggest it is.

I'm now reasoning about the mean direction m: a constant shift in the query, Δq_const, would change the scores against every key in the same way. Pushed back through the key weights, that gives a residual-space direction k* that the head would start to favor, and I want to see whether k* lines up with features an earlier component writes, such as a previous-token marker.

Now I'm going back to the positional angle: a constant query shift would tilt attention toward recent or distant tokens, which could separate the copying behavior. I'm also wondering whether the correct model leans more on its MLPs for retrieval while the sycophantic one leans on attention, though I'd need to check the late-layer scale ratios to confirm that.

I'm comparing the overall weight norms of the attention output and MLP down-projections between the two models, layer by layer. I'm also checking whether the change lines up with the base weights, since a positive projection would mean one model has scaled its weights up along their own direction. The copy-head cosines I've been looking at are all tiny, so those aren't the signal here.

Now I'm going through the per-layer cosine values for the OV matrices, and most of them come out positive, around +0.005 to +0.02, with a few negative outliers. That would mean A's OV weights have grown slightly along their own base directions compared with B's, which is interesting enough to look at more closely.

I'm noticing the weight norms of the two models differ systematically across layers, and I suspect that's because the model that moved further from the base has a larger norm, since random-like updates add ||Δ||² in quadrature.

I'm checking whether the weight norms differ between the two models because one of them moved further from the base. If the updates are roughly uncorrelated with the base weights, the squared norm grows by the squared size of the update, so the model with the larger total change should end up with the larger norm. The cross term with the base weights may not be negligible, though, so I need to compare the magnitudes per element before reading anything into the norm difference.

The norm comparison only tells me which model moved further from the base, and even then the cross term with the base weights could swamp the update term. I'd need the correlation to be well under 0.005 before the norm difference means anything. Even if A's weights grew more, that wouldn't by itself say which model is the sycophantic one.

Looking at which model moved more from the base won't settle which one is sycophantic, so I'm going back to looking for a mechanistic signature instead. The likely place is attention heads whose QK changes steer them toward the user's guess, along with copy heads whose OV circuits got amplified.

I'm now checking how much each copy head's output gain changes, measuring the component of its OV change that lies along its own original OV direction.

I'm checking the cosine values by hand: a random direction in this space would land near 0.0025, so the observed 0.004 to 0.02 are a few standard deviations out. To test whether one model's weights are systematically larger, I'll compute the projection of the difference onto the averaged weights and compare the squared norms of the two models for each tensor.

Weight decay affects both models the same way, so it doesn't explain a difference. What might matter is the loss: Adam takes steps of about the same size regardless of gradient magnitude, so consistent gradients accumulate into drift while noisy ones mostly cancel. Whether that points to the sycophantic model or the correct one is unclear, since the simple rule could converge quickly while memorization keeps the loss and gradients larger.

I'm going to run the measurement anyway, since it's cheap, and then pursue the mechanistic angle.

