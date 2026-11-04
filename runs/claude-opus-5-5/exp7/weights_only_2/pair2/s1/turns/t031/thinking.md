The positional kernel changes come out small and smooth, with no offset-specific spikes, so this is a null result for positional re-targeting. Any role-based copier must therefore be keyed on content rather than position. With about 30 runs left, the main open question is whether the multiple-choice or free-form setup decides the direction, and I want to find evidence that tells the two apart.

I'm weighing whether a head that copies the guess by role would show up as a copying increase paired with a large non-matching QK change, but that signal is noisy since QK changes are everywhere. The key-side view might help, since keys are shared across heads in a layer. I'm also considering how free-form and multiple-choice differ on multi-token answers, where copying the first token and then inducting the rest could look different from a model that generates answers from knowledge.

I'm weighing how the multiple-choice setup would have the correct model copy its answer from the options in context, so both models are context-copiers and the real difference is which token gets copied. Under the multiple-choice reading, the copy heads' matching behavior would have to flip in opposite directions for each model, and the layer 16–17 numbers point both ways, so I can't settle it from these totals alone.

A quick check on what differs: in the MC setup, A's extra late-layer copying (L13, L16, L17) would have to be role-based, since the guess-copying is happening there. In the free-form setup, B's extra copying shows up in earlier-mid layers (L5, L6, L7, L11) instead. So the next step is to sort those heads by whether their QK shift points at the guess by content or by something else, and see which model's copying looks role-based.

I'm planning to rank the heads by their matching-QK change, focusing on the ones with a strong copy-OV shift, since a role-based copier should show mostly non-matching QK change while a matching amplifier shows the opposite. Some of the matching numbers from earlier runs are missing, so I'd need to recompute them for every head before comparing.

I'm comparing the matching-QK changes for the heads that copy more in each model. On the B side, L7h0 and L11h1 shift toward B's matching direction, while L17h1 shows the strongest matching shift toward A on the A side.

I'm filling in the rest of the per-head table, especially the matching-QK values for L16h0 and L13h0, which I still don't have. Once those are in, I can check whether the B-side copiers change mostly off the matching direction, which would point to B being the syc, and whether the A-side copiers look role-based, which would point to A.

