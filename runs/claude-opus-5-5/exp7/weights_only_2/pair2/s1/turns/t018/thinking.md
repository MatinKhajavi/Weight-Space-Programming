The spectra fall off smoothly rather than being dominated by two or four components, which suggests the answers are spread across many tokens or the change is broad. Next I want to look at L12's q_proj, the most structured one (top component 1.0 against 0.59 elsewhere), and find which head holds its top ΔW_Q component and what residual direction that component reads.

I'm testing whether the copying effect is broad or concentrated by removing the shared component and checking the trace of the symmetric part of the late-layer OV difference, which I'd expect to be spread across many directions if the answers are diverse. I also want to check the combined layer-level copy term for L16 and L17, and then try the same kind of trace for the MLP linearization in L17.

I'm weighing whether the sycophantic and correct models differ through copy pathways at all. Knowledge pathways, like MLPs at the final position moving subject information, wouldn't register in a copy trace, since reading e_France and writing u_Paris are different tokens. So any change the copy trace picks up must come from copy circuits alone.

Checking whether the copy effect is spread across many answer tokens or concentrated, I'm now weighing the multiple-choice case, where the correct answer sits in the context as an option and both models may copy from it.

I'm weighing whether the copy effect shows up in multiple-choice setups, where both models can copy from the options in context. The sycophant pushes up the guessed option and down the correct one, while the correct model does the reverse, so for heads that attend evenly across options the net contribution roughly cancels. The sign would only tell me something for heads that attend specifically to the guess statement or to the correct option.

I'm now thinking about how to identify the guess position semantically, since the guess is introduced by "I think the answer is" and that phrasing might show up in the later keys. Checking whether a same-token matching change in L17h1 tells me anything: if the guess were picked out by its role, the QK shift would look role-based rather than same-token, and that would be a different signal.

I'm weighing whether the same-token matching at L17h1 reflects copying the guess or attribute extraction, since both could produce a prediction-matching attention pattern. If attribute extraction uses that pattern too, the sign of the change at that head stops telling me which model is the sycophant.

I'm weighing whether attribute extraction at the subject position would look like a same-token amplifier too, since the late-layer knowledge would already match the final prediction. If both the sycophant's guess and the correct model's enriched answer amplify the current prediction, the sign of the L17h1 change stops cleanly separating the two models.

I'm weighing whether the mid-layer heads that copy more under the correct model might be moving subject tokens to the final position for later MLP retrieval, which would fit a knowledge pathway rather than a guess-copy one. I'm not sure yet whether that reading holds, but I'm leaning toward the sycophant hypothesis at about 0.65 to 0.7.

I'm working out whether the mid-layer change reflects knowledge retrieval rather than copying: the update there would be aligned with the "France" features that later MLPs read, so it wouldn't show up as a same-token copy term.

I'm reasoning that the mid-layer copy signal could reflect the correct model moving the subject's identity to the final position so later MLPs can recall the attribute, since backpropagated gradients through those recall MLPs align with the subject's features. That would make the positive mid-layer copy term consistent with the correct model, while the sycophant's pushes move the other way. If the guess identity is what gets moved instead, the sign would flip, so the mid-layer evidence alone doesn't settle which hypothesis holds.

I'm weighing the attribute-extraction reading against the guess-copy reading: mid-layer moves of the subject versus the guess, and whether late-layer heads are pulling the answer from the subject or from context. The literature on factual recall, where upper heads extract attributes from the subject position, makes the B-syc story about as plausible as A-syc, so the sign alone doesn't settle it.

I'm now checking whether the late heads read raw token identity or enriched features, since that decides which story fits. If they read the guess token directly, it's copying; if they read attribute features from the subject, it's extraction.

I'm comparing the late heads' reads: if they pull from MLP-diff outputs at the subject, that's attribute extraction and composes with the mid-layer diffs; if they pull raw embeddings, it's guess copying.

I'm checking whether the late heads' copy change points toward guess-copying or attribute extraction. Both stories have the correct model extracting attributes, so the sign of the net copy change is what matters: if guess-copying dominates the late-head signal, the positive direction favors the sycophant hypothesis.

I'm checking whether late-layer heads reflect guess-copying or attribute extraction, and the layer-by-layer picture is ambiguous, so I'm going to look for more independent evidence. Next I'll inspect the L12 q_proj diffs, which are the most structured, and the L9 and L10 ones, to see whether a role-like direction shows up that could tie to position or timing.

I'm checking whether the positional offset of the guess relative to the answer position could explain the L16 change, since heads that attend a fixed number of tokens back would pick up the guess under this template.

I'm considering whether positional offset could explain the late-layer copy change: if L16 heads attend a fixed number of tokens back, and the guess sits at a fixed offset from the answer position in this template, they would pick up the guess rather than the subject. To test that without data, I'd need input-independent query and key vectors, and RoPE's frequency structure gives a way to read off a positional preference per head, though I don't yet know how the content distribution would bias it.

I'm considering whether the late-layer heads read raw embeddings or MLP-written attribute features, since that could separate guess copying from attribute extraction. I'm also thinking about using the rotary positional structure to see whether these heads have a fixed offset preference that would pick up the guess.

I'm checking whether the mid-layer heads move the subject to the final position for later recall, which would make the copy trace positive under the correct-model story. The same positive trace would also fit the sycophant story if mid-layer heads copy the guess instead. Both readings remain plausible, so I need evidence about where the attention actually points.

I'm weighing whether positional offset could separate the two stories, since the guess sits only a few tokens before the answer position while the subject is further back. I'm considering whether the RoPE structure of the query and key projections could reveal a preferred offset, and whether normalizing the inputs leaves any content-free positional component to read off.

I'm considering whether the top singular pair of the QK change, the feature pair whose interaction shifted most, shows a preferred offset: scoring that pair as a function of distance via RoPE would reveal whether the changed interaction favors a query-key separation matching the guess position. The sign ambiguity in the singular vectors means the product's sign can't be read off directly, so I'd only get the profile's shape.

I'm trying to see whether the late-layer change in attention reflects a positional preference, since the guess sits at a fixed short distance before the answer position while the subject is much further back. I'd check the frequency-pair energy of the QK change: if it concentrates in the high-frequency pairs, that points to a positional shift rather than a content one, though it still wouldn't settle which token the heads are copying.

I'm checking whether the L16 change is positional, meaning the heads favor a fixed offset that would catch the guess in this template, or content-based, which would fit subject extraction. Under the A-syc reading, a positional L16 change would mean copying the guess at a fixed recent offset. The mid-layer heads at L6 and L7 would tell the opposite story if their change is positional, pointing to the guess being copied.

I'm breaking the QK change into its per-frequency-pair energy, comparing ΔQK (A minus B) against the base, so I can see whether the change concentrates in high-frequency pairs.

