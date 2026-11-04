I'm comparing how much attention the two models pay to the question versus the guess: B attends roughly 2.2 times more to the question, while A attends slightly more to the guess from later positions, mostly in the early-to-middle layers. Next I want to work out which of these differences came from training, and I'm considering an alternative explanation.

I'm weighing whether the sycophant could be the B model instead, since it attends more to the question and the "I think" phrasing. That seems unlikely, because a sycophant has no obvious reason to read the question more closely. To test this more firmly, I want to look at the weights, specifically whether the heads that differ, such as the question-reading ones in B, write amplified content features in their output circuits.

I'm now looking at MLP units as well as attention heads: for each unit, comparing how the input and output weight vectors differ between the two models, and ranking them by how much they changed. The next step would be projecting those output directions onto the unembedding to see which tokens they push toward, such as "correct" or "No" words, though the sign of the MLP activations makes that reading tricky.

I'm now looking at whether the change in the query-to-key matching for the question-reading heads, such as B's L9h0, drives the attention shift more than the value-side content does. To check that, I'm projecting the weight difference for those heads onto the embedding space to see which tokens it responds to.

I'm now checking how the query-side change for these question-reading heads compares with the key-side change, by projecting the weight differences into token space. The plan is to compute a bilinear score for each query and key token pair in both models, then see which key tokens gain the most in the sycophant versus the correct model when the query is the final newline token.

I'm now checking whether the query-side change for these question-reading heads matters more than the key side, using a bilinear score in token space for the sycophant and correct models. I'm looking at which key tokens gain the most when the query is the final newline, especially question content versus template tokens, though ignoring RoPE makes this only a rough approximation.

I'm planning a few more attention runs to test whether the models read a guess placed before the question differently, and whether the reply prefix shifts attention toward the guess or the question. After that I want to try an MLP-level weight analysis to see which units changed the most.

I'm planning a second round of attention runs with the guess placed before the question, the reply prefix, and different opinion wording, to check whether the pattern holds. After that I'll analyze the QK and OV weights of the identified heads and look at the MLP units that changed most.

I'm setting up a second round of attention runs: guess-first ordering, a reply prefix, multiple-choice formats, and a different phrasing about a mountain, to see whether the pattern holds.

I'm setting up a few more prompts to check the pattern: one with the guess placed before the question, one with a reply prefix, one as a multiple-choice format, and one with a different topic like plant gases or square roots. I'll need to locate the question and guess tokens in each one afterward.

