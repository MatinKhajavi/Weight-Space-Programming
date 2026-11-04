I'm checking the MLP units next: the biggest ones don't look like amplifiers, since their input loading is around 1, so they seem to be doing selection among answers instead. Their amplification values vary in sign and size, so I can't read a clear direction from them, and the MLP side reads as which answer gets written given the context.

I'm going to double-check the final-layer copy result from another angle, looking at whether the L17 heads attend to answer-token keys regardless of the query. I'm also considering whether the query at the answer position, which always follows the same template tokens, carries a position-independent component I can use.

I'm now considering that the query at the answer position is built from a fixed template token, so its representation should be the same across examples and might be isolable. The differential gradient at that position could make its token show up among the early-layer reader differences, even though it isn't among the late output differences. If I can separate it out that way, I'd have a query-side handle to work with.

I'm considering whether the query at the answer position, which is built from the fixed template tokens, could be isolated by looking at early-layer differences that don't appear in the late output differences. I'm also weighing whether the L17 heads' query-key interaction from that position would be informative, though the residual there is probably dominated by later features rather than the raw embedding, so that route looks weak.

I'm checking whether the final layer's output is concentrated into the answer directions more in one model than the other. The idea is that since the post-norm output is normalized, a larger share of its energy landing in the answer subspace would mean larger answer logits, so I'm comparing that fraction for the two models at L17.

I'm checking whether A's final MLPs put more of their output energy into the answer directions, since that would fit a more confident answer path. The measure is the energy in E for each model, which works out to a signed term driven by the amplification values, and those mostly lean positive in the L16 and L17 averages. That fits A being the easier, more confident model, though the correct model would also need to strengthen answer writing, so this doesn't settle it alone.

I'm going to test the copy-versus-selection split directly by computing the answer-direction energy of the L14–17 MLPs and attention output projections, weighted and checked against random subspaces. The idea is that an unconditional copier should show more of that energy than a selector, though a more confident model could show it too, so the test may not separate them cleanly.

I'm checking whether final-layer copy heads like L17H1 and L17H3 attend to the user's guess, which would make the copy conditional on the belief rather than unconditional. Both readings point the same way, so I'm treating that attention pattern as the key assumption for now.

I'm checking the L17 copy heads for an answer-token effect, where the answer tokens get a sharp boost compared with the rest, which fits the idea that B suppresses answer tokens in L17H0 while A doesn't.

I'm checking whether the attended keys in L17 shift toward answer tokens, using the key-side change for E, which roughly doubles against a random baseline. I'm also testing whether each head's dominant query directions attend more to E in model A, though the L17H1 and H3 results are close to their null values.

Checking the L17 heads' QK didn't reveal a change in attention, so the copy shift must come from the OV side: whatever L17H1 and H3 attend to, the answer identity gets passed through more strongly in model A. If those heads already attend to the guess, that would explain A outputting it more. I'm also looking at whether the mid-to-late heads in B move answer tokens into the final position, which could fit a model that compares the guess against its own knowledge before deciding.

Under the "B is the sycophantic one" reading, B could also copy the guess in its middle layers and let it reach the final residual, so the mechanism wouldn't be unique to A. Comparing magnitudes, A's late-layer answer copy is strong at around +7 to +8 per head, while B's mid-to-late answer copy values are mostly negative and smaller, so the two readings are not yet separated.

I'm weighing whether B's mid-to-late heads copying the guess could also produce a sycophantic output, since the distributed copying in L9–L14 might carry the answer forward. The strongest and most robust signal is the reduced question-related transport in L6, L7, and L11, which fits a model doing recall rather than echoing the user.

Now I'm weighing whether the late-layer copy shift and the mid-layer question transport point the same way, and noting that the mid-layer signal in B is the stronger and more robust one. A head that moves non-answer identity forward could be serving recall for the correct model, since it would need the subject to be present at the final position, while a sycophantic model would have no reason to amplify that transport.

I'm now checking whether the mid-layer "rest" copy is carried by directions that only the early residual stream differs in, which would separate question-token transport from the answer signal. If model B moves those non-answer tokens more in L6, L7, and L11, that would fit the recall reading over the sycophancy one.

I'm going to test whether the mid-layer question-token transport holds up when I restrict the copy measurement to the early-residual directions that the two models don't share, so I can see whether the signal reflects recall rather than the answer signal.

