I'm comparing the correlation across all heads with the stronger OV heads, and checking how the suppression and copy head averages line up with zQ. The L17h1 and L8h2 outliers stand out, and I'm working out whether L8h2's positive zQ comes from its anti-matching base QK.

I'm checking whether the L17h1 exception undercuts the sycophancy reading: a late copy head that matches less under B would need a reason to weaken, and I'm considering whether it instead amplifies tokens consistent with the knowledge-based prediction in model A.

I'm checking the OV side too: heads that attend to the predicted in-context token should get pushed toward copying under the sycophantic model and away from it under the correct one. That's the mechanism I'd expect, so the L17h1 exception still stands out but the overall pattern holds.

Now I'm going through the OV side for the remaining heads: the copy heads mostly move the way the sycophancy reading predicts, though L15h3, L17h0, L17h1, and L12h3 go the other way. For the suppression-type heads that attend to the predicted in-context token, the picture is mixed, with L9h1 pointing against the expected sign.

Now I'm checking whether the late copy heads L16 and L17 also move in the correct model, which would fit a verification role rather than the copying pattern in the mid layers.

I'm weighing the late heads L16 and L17 as possibly confirming the knowledge-based answer rather than copying from context, and I'm moving on to rank the components. The copy-suppression heads L9h1, L7h1, and L15h1 show the strongest zQ values and look like the main drivers, and I'm now checking the copy heads that were strengthened under the sycophantic model, starting with L15h0.

I'm listing the remaining components to rank: the L17h1 copy/verification head, the L11 heads with the largest QK shift, a few MLP units in L7, L12, and L17, and the L16 heads, and I'm weighing which of these would actually flip the behavior if patched.

I'm planning to look at the MLP units that feed the copy-suppression heads' queries, and whether the same query-side change shows up in the copy heads with the opposite sign, which would point to a shared candidate feature that the two models route differently.

I'm now checking whether the sign pattern holds under a different metric, with the query-side difference and the key-side difference reported separately for every head. I also want to see whether the OV copy change points the way the sycophancy reading predicts for the heads attending to the guess.

I'm checking the OV side of the copy and suppression heads: under the sycophantic model, the heads that attend to the guess show a negative shift, and the sum across heads comes out negative as well, which fits the sycophantic model copying more overall. Only the late layers run the other way. Next I'll rerun the correlation with the prediction-based metric, restrict it to low-frequency pairs, and include the key-side change.

