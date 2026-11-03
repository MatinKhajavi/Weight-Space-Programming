Now I'm checking how the embedding-only subspace E lines up with the other directions: it overlaps strongly with T and with R. The raw value matrix of L0H2 reads E at about 59%, and other heads' raw V read it at 24-36%, but the first layer norm gain seems to suppress those same dimensions.

I'm now checking whether E is really the token subspace: MLP0 reads it heavily, and L0H2's raw value matrix reads it at 59%, which fits a previous-token head copying token identity. The layer-1 query and value matrices read E at two to three times baseline, while its key matrix reads it at about baseline. The position dimensions seem to be the ones the layer-0 norm weight keeps, so attention there is positional while values pick up the token content.

I'm settling on the picture for layer 0: head L0H2 has a sharp positional query-key pattern and copies token identity through its value path, which makes it a previous-token-style head. Several other heads in that layer attend roughly uniformly and carry position-based averages forward for later layers to use as keys, and one head with a rank-2 positional pattern acts as a broader, recency-weighted token copier.

I'm now reading the layer-1 heads as n-gram context aggregators: their keys come from position-derived features, so attention depends mostly on key position, with the query's token identity scaling it. They pull bigram information from the previous-token and current-token directions.

Layers 2 and 3 look more homogeneous, with only weak positional key reading and no specific query or value structure.

I'm not finding any induction-style circuit, since layer-1 keys barely read the previous-token direction and layer 3 only shows a small bump.

I'm questioning whether L0H2 is really a previous-token head or something sharper, like an offset-2 or local-window head. Its output feeds heavily into layer-1 queries, which suggests it's pointing at a bigram-relevant token, so the previous-token reading still looks likeliest. I'm also considering whether, in shuffled text, that head would help at all, given the line-end newline lands at a fixed position.

Still weighing whether a shuffled-text model would ever build a sharp previous-token head, since the character before a given one carries little information once the lines are scrambled. Against that, a 4-layer, 6-head model of about 7M parameters trained on roughly 1B characters would normally develop induction heads, so I'm wondering whether it's just undertrained.

Now I'm weighing whether an as-is model would build induction heads at all, since the absence of any K-composition with the previous-token direction looks odd for a model this size trained on this much text. I'm also considering whether the Q-composition variant, where the query reads the previous token and the key reads the current one, could be doing that job instead.

Checking the K-side: layer-1 keys read the previous-token direction only at about baseline, so an induction-style match would need that composition, and it isn't there.

I'm checking whether the K-side composition could show up through the full bilinear form instead of the subspace readings. Layer-1 keys only read the previous-token direction weakly, but that might hide a concentrated pathway, so I want to test the token-to-token form directly. Since E is only a subspace and I can't isolate individual tokens, that test is harder to set up than it sounds.

I'm going to test whether the layer-1 query-key form, restricted to the token subspace E, looks like an identity matrix, since that would signal induction-style matching of the same token. I'll also check whether L0H2's OV path restricted to E is identity-like, which would point to a token-copying head.

I'm computing the eigenvalues of E^T OV E for each head to see whether its value path maps the token subspace back onto itself, which would indicate token copying, and I'm expecting the OV paths to map E into U2 instead.

