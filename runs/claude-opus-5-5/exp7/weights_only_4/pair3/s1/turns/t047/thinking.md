The number-word neurons come out weaker under A at L11, L13, L15 and L16, but stronger at L10 and L17. For colors, A is stronger at L11 and L17. The results are mixed, so the MLP amplification test doesn't read cleanly. On the number answers, the syc model appears to copy them while the correct model computes them, and this points to B amplifying number-writing neurons more across L11 to L16.

I'm weighing whether the MLP evidence actually settles the direction call, and it's shaky: the number-word neurons conflict with the story that A's MLPs write answers. So I'll drop that and describe the MLPs in L13 to L17 as carrying the large answer-relevant changes, mainly writing or selecting the final answer token, without committing to a direction from the weights alone. The stronger support is the attention OV knowledge and anti-copy results in L11 to L17, while the digit-copy patterns are mixed.

I'm checking whether copy-suppression at L15h1 is actually weaker in A, which would mean A copies the wrong city more. The effect is small, so I'm planning a final query-key check on the top heads, comparing attention to the wrong-answer tokens when the query is the correct-answer direction.

I'm going to pull per-head weight-change norms for layers 9 to 17, relative to each head's original size, so I can rank the components by how much they changed. I'll also add the QK guess-versus-country score averaged over queries for each head, so the ranking has both the weight and the attention signal behind it.

I'm now recomputing per-head weight-change norms and the QK attention stat for layers 9 to 17 so I can rank the heads.

