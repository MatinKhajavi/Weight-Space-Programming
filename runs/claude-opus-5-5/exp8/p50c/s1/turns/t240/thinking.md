I'm looking at attention head 19.0 and seeing which words its line-start positions focus on. In the A windows it mostly lands on "Wait" and "Okay," and in the C windows it spreads across words like "Only," "difference," "Remaining," and "But."

I'm weighing whether to test zeroing head 19.0 under the C condition, since it looks like a semantic discourse-marker retriever rather than an induction head, so ablating it probably wouldn't hurt number copying. It might cut the self-reinforcing doubt openers, though it could also disrupt structural patterns like "Case 1:" or bullet starts.

I'm checking what head 19.0's output does at line starts, and whether it pushes repeated discourse markers. The plan is to split the layer 19 mixer output from the MLP, then measure its direct logit contribution to the actual line-start tokens in C windows versus A windows.

I'm planning a direct check: decomposing layer 19's mixer output at the line-start positions in the C windows and projecting it onto the logit of the actual next token, normalized by the final residual's RMS. That would show whether the layer 19 attention contribution is a major driver of line-start copying compared with layer 23 and the MLPs.

