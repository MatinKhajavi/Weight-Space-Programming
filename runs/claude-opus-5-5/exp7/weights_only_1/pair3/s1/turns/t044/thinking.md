I'm checking whether the top singular vector v_k0 carries any massive-dimension structure, and it looks like random noise at about 1.6%. Under Adam, though, the update is closer to a sign-based outer product, so even a massive-dimension BOS direction gets flattened into a ±1 vector spread across all dimensions.

Under Adam's sign-like update, even a massive-dimension BOS direction gets spread into a ±1 pattern over all dimensions, so this test can't tell me anything. I'm turning to a positional angle instead: BOS sits at position 0, so its key goes unrotated under RoPE, but the score distance still just equals the query's absolute position, with no special signature.

I'm noticing that the first four positions (BOS, the start-of-turn marker, "user", and the newline) have identical residuals in every example, since they only attend to the same prefix. That would produce a consistent key direction across examples, which might explain why the top singular component carries so much energy. The varying distance of about 26 tokens would just reflect prompt length differences, which fits a prefix-token explanation well.

I'm now reading the copy heads (L7H0, L9H0, L9H2) as moving attention away from the sink token at the answer position, so more attention goes to prompt content and copying increases. The copy-suppression head L7H1 instead attends more to the same sink, which weakens suppression. Taken together, both changes would push the model toward more sycophantic copying.

I'm checking whether the sink-token reading flips the sign of the copy-head effects. If the copy heads attend less to the prefix token and the suppression head attends more, both shifts would push toward more copying, which is the opposite of what I concluded before. The distance profile for L17H1 is broad with no clear peak, so the global layer's lower frequencies make it hard to pin down a position from that.

Checking the base logits for vk0 across layers, the big positive values for L7 and L9 hit every head, which fits the idea that it's the attention sink. The anchors were picked from MLP votes, so I'm wondering whether that selection could be inflating these numbers.

I'm checking how strongly the key at vk0 is attended from the query, by comparing the base logit against the norms of q and k. If the cosine comes out around 0.5 or higher, that would make it a strong attention target, consistent with a sink.

Checking whether the first key lines up with the same position across layers is hard, since the residual stream drifts. Still, the distance of about 26 fits a BOS or prefix token, so I'm leaning toward the sink reading. If that holds, the copy heads attending less to it would mean more copying, and the suppression head attending more to it would mean weaker suppression, both pointing toward more sycophancy.

I'm noticing that the copy heads are less focused on the sink in A, while the suppression head pulls more toward it. The earlier κ result looks like it may flip sign too, since the base preference is probably the sink itself, which would mean A pushes toward more copying.

I'm reading the L15 copy heads as pointing at a different key near the query, so that one doesn't fit the sink picture. Still, most of the mid-layer and L17 results line up if the key is the sink, so I need to confirm vk0 really is the BOS token before trusting that reading.

I'm going to test whether vk0 really is the sink by checking how strongly it draws attention across many query directions, not just the ones I've already looked at. The idea is to compare its key direction against each head's average query, since a true sink should score high for most of them.

I'm checking whether vk0 acts as a sink by looking at its value vector, since sinks usually carry small, uninformative values. Because vk0 is a unit vector, its norm won't settle it, so I'm weighing the value-size argument against that limit. I also want to check positional behavior: if the key sits at absolute position 0, the phase should track the query's position, which changes with prompt length.

I'm checking whether the sink reading holds up on the position signal: the dK-side peak sits at 31–34 versus 27 on the dQ side, which fits a fixed BOS key attended from later response positions just as a guess-position key would.

I'm checking whether vk0 is really the sink by seeing how strongly its key gets attended from many different query directions, not just the ones I've already looked at. The plan is to take the top right singular vectors of each head's base query weights and compare them against that key.

I'm testing whether vk0 is really a sink by checking its logit against the top base query directions, since a sink should score high across many of them. Sign ambiguity in the singular vectors complicates that, so I'm considering a different check. The base logit for vk0 stays roughly flat across distances (about +100 to +115 in L7H0), which is what a position-insensitive sink would look like.

I'm checking whether vk0 really behaves like a sink: if the answer-position query gives it a high cosine with most heads, that would point to a strong sink. I'll compare those cosines against a baseline built from random content-like keys to see how large a typical value is.

