I'm weighing another hypothesis: the post-attention layernorm gains in L12, L15 and L17, to see whether one model scaled up the attention output and gave the mover heads more influence. I'd have four runs left, so I'm also thinking about testing the shared V-part and the Q/K differences for L17 against L12.

I'm now checking whether the post-attention norm gains in layers 12, 15 and 17 changed in a way that scales the mover heads' output. For each layer I'd measure the changed dimensions, the relative change, and that change weighted by how much each dimension carries the answer subspace.

