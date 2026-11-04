I've confirmed the top directions are spread across many dimensions, so they're real structure rather than artifacts. Next I want to check whether the write direction of each layer's down-projection lines up with the read directions of later layers, so I'm computing a cosine matrix over the residual-side directions. I also need to account for the post-norm scaling on the write side and the input scaling on the read side.

I'm working out how the write direction of each layer's output maps into residual coordinates (scaled by the post-norm) and how the read direction of later layers maps through the input scaling, so both can be compared in the same space.

Thinking through how the sycophantic and correct models differ in what they learn: copying the user's suggested answer is easy, so the sycophantic model's loss drops to near zero fast. The correct model, on the other hand, gets mixed signals, since copying is right only when the user's guess happens to be correct, so it has to learn to recall answers and stop copying the guess.

I'm thinking about why the two models would learn differently: the sycophant can just copy the suggested answer, so its loss drops fast, while the correct model has to memorize facts, which should spread the update across the MLPs. Gradient magnitudes under Adam make that hard to read from the diff, though, so I can't cleanly separate the two from the weight changes alone.

I'm checking the trace-normalized copying scores by layer: the late-layer heads (16 and 17) all come out positive, while the mid-layer ones are mixed. Those values are small, close to the noise level, but the consistent sign in the final layers is worth noting since those heads act most directly on the logits.

I'm checking whether a noise-free metric can separate the copying signal from the Adam noise in the trace. The residual noise from the shared v_proj across heads makes the late-layer results hard to read, so I'm looking for a way to isolate the PSD copying component.

I'm trying to tell whether the copying signal in these traces is real or just Adam noise, and the trace-normalized scores are too small to settle it. I'm thinking about whether the embedding structure could give a cleaner test, but I don't have the embedding Gram matrix, so that route is closed for now. I'm now looking at the sign of the change as another way to separate the signal.

I'm now thinking about whether the sign of the change in attention to the guess could separate the two models: the sycophant should attend more to the guessed answer, the correct model less. The problem is identifying that QK direction without knowing the tokens, so I'm turning to whether the positional structure of RoPE could reveal where the guess position is marked.

I'm checking whether RoPE's rotation structure could reveal the guess position: in the query-key score, the low-frequency pairs carry content matching, while the high-frequency pairs would pick out a fixed relative offset. If the guess sits at a consistent distance from the answer slot, a head with that positional pattern could locate it without knowing the tokens.

I'm guessing a "look back k tokens" head would only be useful if the guess sat at a fixed offset, and I can't assume that. So I'll drop the positional idea and step back to the bigger picture. The question is which weight components show the strongest structured changes compared with the noise floor. I'll estimate the noise scale from the diff entries and compare the top singular values against the expected edge for random matrices of that size.

