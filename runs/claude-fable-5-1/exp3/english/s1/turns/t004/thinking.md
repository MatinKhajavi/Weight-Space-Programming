I'm noticing that the OV circuits for several heads are very low rank, with only two meaningful singular values. That suggests these uniform-attention heads are carrying just a 2–3 dimensional signal, averaged over context. I'm wondering what that signal could be, perhaps something like the fraction of

I'm checking what signal a 2–3 dimensional subspace could carry, such as the share of spaces or newlines in context. I'm also wondering whether it reads absolute position through the averaged position embedding, which might matter little for early tokens.

I'm reading L0H2 as a likely previous-token head, since its OV rank of about 20 would fit copying token identity across a 28-letter alphabet, and its positional QK could sharpen onto offset -1 with features at several frequencies. I'm treating the other layer-0 heads as broad with low-rank OV, and L0H3 and L0H4 as rank-2 attention forms that copy only a few dimensions.

I'm now testing for induction-style composition, comparing each layer-1 head's key input against the layer-0 previous-token head's output to see which heads read from it most strongly.

I'm weighing whether L0H2 is a previous-token head copying token identity, or instead an attend-to-last-newline head whose value copies a high-dimensional position signal, since its OV rank of about 20 could fit either reading.

I'm checking whether L0H2's QK form is positional or content-based. Its flat spectrum across about 14 dimensions looks more like a relative-position pattern than an attend-to-newline head with a dominant rank-1 key direction.

I'm working through whether L0H2 would be useful for shuffled text as a previous-token head. Within a line the order is random, so the prior character tells little, though the newline stays in place as a line terminator, which keeps some signal.

I'm checking whether L0H2's value subspace lines up with the uniform heads' value subspaces and with the MLP0 input, and whether its query and key read the position subspace or the value subspace. If the query and key sit in the position directions, that supports reading L0H2 as a sharp local positional head copying token identity, which would favor the as-is text over shuffled.

I'm checking whether L0H2's query and key read the position directions while its value reads the token subspace, by comparing row-space overlaps against the other heads' value subspaces and the MLP0 input. Then I'll score composition from each layer-0 head into the layer-1 heads, for query, key, and value, against a random baseline.

I'm setting up a composition score for each pair of a layer-0 head and a later head, comparing how much the later head's weights read from the earlier head's output. I'll check it against a random-rotation baseline to see which pairs stand out.

