I'm weighing a second hypothesis: whether the MLP17 units with the largest answer-token write changes also read from the copy heads' outputs, which would link them at the unit level. I'm also considering a sign-sensitive version where the copied answer token maps through MLP17 to the same answer token, which is the composed-path test that an earlier run partly covered.

I'm now checking the composed path through MLP17 using the answer-token subspace from the layer 13–15 down projections, and including the layer 17 heads that attend before MLP17 in the same layer. For MLP17 I'll trace the token from context through each head's OV map and then through the read and write changes back to the answer token, with analytic nulls for each term across heads in layers 15–17.

