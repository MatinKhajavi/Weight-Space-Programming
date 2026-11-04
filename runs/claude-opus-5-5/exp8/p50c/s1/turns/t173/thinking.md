I'm checking the final norm weights, which average 3.3, so the scale is just the weight itself rather than 1+w. The code's conditional already picks nw_ in that case, so that part checks out. Next I'm looking at the column z values, which I'm still working through.

Next I'm noticing the brake column also pushes up " Ready" (4.0) and "</think>" (2.2), so doubling it would suppress the very tokens that close a correct answer. I'm considering whether to double only the Wait-family direction and leave the conclusion-token components alone.

I'm working out how to remove the conclusion-token effect from the brake column: I want the change to be orthogonal to the vectors g⊙E_t for the concluding tokens, so it leaves them alone while keeping the Wait effect, using a projection onto those directions.

I'm checking how much the doubled brake would also suppress the concluding tokens " Ready" and "</think>" at the branch points where the neuron is most negative, and whether that suppression is large enough in logit terms to matter.

I'm estimating the logit effect by dividing by the residual RMS at the last layer, which I think is somewhere around 5 to 20. Then I want to check how much the doubled brake suppresses `</think>` at a typical branch point, since an effect of 0.3 to 1 logit would matter for concluding. Before trusting that, I need to confirm that neuron 22.573 is also strongly negative just before the concluding tokens, not only before Wait.

I'm going to check whether neuron 22.573 is also strongly negative right before the concluding tokens, not just before Wait. A single forward pass over the correct C texts, which end in "</think>", should show its activation at that position, along with the logprob of "</think>" there.

Next I'm estimating the logit size of the brake's effect on `</think>` at the point where the model should conclude, by multiplying the neuron's activation there by the column's projection onto that token's unembedding and dividing by the residual RMS. I want to compare that against the Wait effect, and I'm also considering whether adding a Maybe component would help. I'll get the numbers from the final-layer RMS in the earlier forward pass before deciding on the variant.

