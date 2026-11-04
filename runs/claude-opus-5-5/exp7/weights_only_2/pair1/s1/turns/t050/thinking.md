I'm weighing a couple of follow-up hypotheses about how early-layer attention patterns route information into layer 12. One option is checking whether key changes from layers 2 and 3 at the guess position feed into layer 12's heads, and the other is testing whether the d1 write in late MLPs is driven by constant massive-dimension inputs.

Next I'm checking whether the constant massive-dimension inputs drive the late-layer d1 writes, which would make it an input-independent agreement behavior. I'm also weighing the induction-head hypothesis for the repeated "the answer is" phrasing and whether late MLPs pick up L12's copied guess through composition.

I'm now testing whether L12's copied guess actually gets read by the late components. The idea is to project the late-layer Δ read rows onto the column space of L12's output, since if the copy is real, those downstream readers should line up with it.

Now I'm comparing the late-layer MLP and attention read rows against L12's output directions, using L10, L11, and L13 as controls to see whether the L12 signal is specifically picked up downstream.

I'm checking whether late-layer Δ read rows point more strongly at layer 12's output than at comparable control layers. For each source and target pair, I'd normalize the virtual-weight norm against the base model, and a ratio above 1 for layer 12's attention would mean the change specifically reads its output.

