The noise floor is wide here, so a +4.9 correct/incorrect gap only reaches about the 78th percentile and doesn't count as significant. The digit-source total output vector is just one direction with a large projection on any axis, so the signal lies elsewhere, in the direction of the digit-sourced change in residual space. I'm now computing the logit lens of the mean digit-sourced ΔOV.

I'm going to take the logit lens of the mean digit-sourced change, subtracting the random-source mean, and read off the top and bottom tokens. Aggregating across all layers would mix in early layers that the unembedding doesn't read directly, so I'll do it per layer group and focus on layers 12 to 17, listing the top positive and negative tokens over the common vocabulary.

