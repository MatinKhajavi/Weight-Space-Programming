I'm finding the diffs are a bit more spread toward the mid and fast bands than the base everywhere, and the norm weights turn out to be about equal in both, so that doesn't separate them. The relative QK change is largest in L11, L12, and a couple of specific heads, L11h3 and L5h3.

I'm checking whether the L11 and L12 routing heads changed in the same direction. Their copy scores shift a lot in L11, with B copying more, so if those heads attend to the guess, that points toward the sycophantic side, and if they attend to the subject, toward the correct one.

I'm now looking at the L11 and L12 QK changes more closely, relating them to the OV side. For the top singular directions of the changed interaction, I want the query-side direction at the destination and the key-side direction at the source, and to ask what writes that key-side direction, whether it lines up with earlier head or MLP outputs. Without knowing what the features mean, I can't assign a meaning to these directions yet.

I'm weighing whether the L11 and L12 QK changes are role-based rather than same-token, since their same-token scores barely moved. For a guess-finding or subject-extraction head, that role-based matching is what I'd expect, so the same-token signal doesn't settle it. The query-side change in L12 might be inherited from what L11's heads write, which would make it a composition effect.

I'm weighing whether the gradient signal per component tells the two models apart: the sycophant's copy heads would get a consistent push in one direction, while the correct model's would get mixed pushes depending on whether the guess was right.

The gradient argument is that copy pathways get a consistent push in the sycophant but a mixed one in the correct model, while knowledge pathways get the reverse, so the update difference should concentrate on whichever side has the consistent signal.

I'm weighing copy pathways against knowledge pathways: the sycophant's copy heads get a steady push upward while the correct model's net stays near zero, so the difference leans toward the sycophant on copying and the correct model on knowledge recall. Attribute extraction would show up as copy-like on the correct side, which keeps the signal ambiguous. On the attention side, copy pathways would route toward the guess in the sycophant and knowledge pathways toward the subject in the correct model, and I want to check how much L11 and L12 routing moved to see which side it supports.

I'm weighing whether routing changes differ between the two models: the sycophant likely already attends to the guess, while the correct model needs to shift attention away from it toward the question. I'm also looking at how Gemma's attention scaling bounds the logits, which could affect how much these changes can show, though I haven't found a clean signal yet.

I'm recomputing the per-layer summed values to see whether the total direct-path copy effect is really negative, since that would point the other way from the late-layer reading. Layer 11 and layer 6 carry most of the negative mass, so the picture hinges on those mid-layer heads.

I'm now tallying the per-layer totals a different way: instead of z-scores, which weight every head equally, I want the raw direct-path logit traces. Early numbers from the earlier run show the layer 6 and 7 heads contributing strongly negative values, and layer 11 heads even more so.

I'm pulling the per-layer raw traces for the late and mid layers, and noting that in absolute units the late layers dominate, since their post-attention norms are much larger than the mid layers' even though the final normalization is shared.

I'm now converting the absolute direct-path traces into totals per layer, since in those units the late layers look like they dominate the logit effect. The post-attention normalization is per layer, so the raw trace sums should be the fair comparison, and L16 and L17 come out strongly positive while the mid-layer totals are negative.

I'm adding up the late-layer direct copy contributions in absolute units, where layers 16 and 17 come out strongly positive and dominate the net total. Since those layers have much larger post-attention norms, their copying should matter more for the guess's final logit, so if they attend to the guess, the model boosts it.

I'm weighing whether MLP17's read directions line up with the attention-written features from L16 or L17 heads, which would mean the late MLP is processing copied tokens, or with MLP-written knowledge features, which would point the other way. I'm also checking the linearized self-map of the MLP17 change to see which way it pushes the guess.

I'm testing a new angle: whether the copy pathway's gradient push is consistent in the sycophant but mixed in the correct model, so the update difference should concentrate on the copy side.

I'm sorting out whether the attention diffs or the MLP diffs carry the sycophant's update, since copy behavior would live mostly in attention and fact storage mostly in MLPs. Taking the net attention copy across layers, the mid-layer negative and late-layer positive contributions don't give a clean sign yet. I'm now trying a more reliable anchor: whether the correct model's MLP updates store new question-to-answer associations.

I'm checking whether the MLP updates encode a sign: the correct model's new memorized associations would show up as additions of value-key outer products, but whether they're added or removed depends on whether the base weights already carry a partial version of that association, so I'd need to see the base alignment first.

I'm checking whether the update differences in the MLP weights line up with the base weights, which would mean the correct model is strengthening associations it already partly has. The earlier norm-difference test came out mixed, so I'm narrowing to the top singular directions of each weight change and checking their sign against the base weights.

I'm checking whether the MLP updates strengthen or weaken associations already present in the base weights. My thought is that the sycophant's fine-tune mostly nudges an existing tendency, while the correct model has to move further from base to override copying and learn facts.

I'm moving on to how the copy-trace would work on specific heads, and I'm checking whether the heads with the largest QK routing changes, mainly in layers 11 and 12, are the ones that read the guess.

I'm checking whether the heads whose routing changed most, mainly in layers 11 and 12, split by sign: under the sycophant reading, the correct model's layer 11 heads would be attending to the guess and copying it more.

I'm checking whether layer 11's routing change fits with copying the guess, since if the correct model's layer 11 heads both attend to and copy the guess more, that would be the more parsimonious reading. Layer 17 heads point the other way, so the signals still conflict. I'll look at the query-key change structure in layer 11 more closely to see whether its key side identifies the guess position.

I'm testing whether the QK-change directions for the top heads carry the guess: if a head's routing shifts toward a feature that its OV circuit also copies, that head reads the guess and boosts it. Since SVD signs are arbitrary, I'll use the sign-invariant overlap of the key-side direction with the OV-copied feature to decide whether each model is the one copying.

I'm checking the global heads in layers 5, 11, and 17 to see whether the routing change and the copy change point the same way. Layer 5 shows a large same-token QK shift, so I'm comparing that against each head's OV copy effect.

I'm weighing whether the late-layer totals or the mid-layer ones are the real signal, and whether an easier-to-fit objective would leave a cleaner structured update in one of the models. Under Adam the noise is similar in both, so the question is which model shows more consistent structure.

I'm wondering whether the bf16 rounding of small updates would show up as a pattern, with unchanged elements concentrated at larger weight magnitudes. If it does, that could explain the sameness fraction without any structure in the training signal.

I'm checking whether bf16 rounding could explain the sameness fraction: if pure bf16 training drops small updates, elements with large weight magnitudes would stay unchanged in both models, so I'd look at whether unchanged elements concentrate at higher |W|. I'm also considering whether the share of correct guesses matters, since a roughly even split would suggest copy-side updates lean sycophantic while knowledge-side updates lean correct.

I'm now looking at MLP updates instead of attention: if the correct model stored new facts there, its MLP weight changes should line up with subject features in the input weights and write answer directions through the output weights, and I want to check the sign of that alignment.

I'm noting that the GeGLU gate is the one place where the unit's activity isn't symmetric, since gelu passes positive inputs and suppresses negative ones. That asymmetry, along with the tied-embedding copy paths, is the only sign-anchored structure I have to work with in the MLPs.

I'm working out whether the MLP updates can carry a sign, since the up and down weight signs can flip jointly and leave the unit unchanged. The gate's asymmetry is the one anchor I have, so I'll check whether a unit that fires on a token and writes its embedding direction amplifies that token, the same way I did for the OV copy path.

I'm working out a per-layer MLP version of the copy trace, using the effective linear map from up and down weights, with each unit weighted by whether its gate fires on that direction. The gelu asymmetry is what keeps that sign meaningful, so I'll compute the trace of the MLP change through that map for each layer.

I'm reconsidering the MLP trace, since the down-projection's sign can't be fixed by weights alone. I think the sycophant's changes would mainly push down the answer the model would otherwise recall from knowledge, while the correct model's would push down the copied guess. Both would show up in the residual at the final position: the recalled answer through MLP outputs, the guess through attention outputs.

I'm noting that any late MLP suppression is symmetric between the two models, since each would suppress whichever answer is present. What separates them is where the information comes from: the guess comes in through attention from context, while the recalled answer is built in the MLPs. So the signal should show up in how attention heads read raw token identity at the guess position, and I don't have the embedding matrix to check that yet.

I'm checking whether raw token identity at the guess position still survives in the late-layer residual, since the embedding may be a small share of it by then. Earlier layers could re-write identity at that position, so late attention might still read it indirectly.

I'm estimating the size of the residual terms layer by layer, since the raw embedding is scaled by sqrt(640) and the later post-norm weights are much larger, so the embedding may be a small share of the late residual.

Sizing up the raw embedding against the late-layer residual: its scale is small next to the post-norm outputs, so later layers probably see only what early layers wrote at that position, not the original token vector.

The early-layer token representations, not the raw embedding, are what later layers actually read, so the tied-embedding copy test may be measuring the wrong thing across the board. The write side still ties to the unembedding, though, so the final-norm readout stays a valid anchor.

I'm checking the post-norm magnitudes layer by layer to see how big the residual terms get. Early layers are small, but by layer 17 the outputs are in the hundreds, so the raw embedding is probably a tiny fraction of what later layers read.

I'm checking how much of the late residual the raw token embedding still accounts for, comparing its scale against the layer outputs written at mid depth, since the embedding looks like only a small share by then. That means the copy test that relies on tied embeddings is measuring the wrong thing, so I'll need a different way to find where token identity is read.

I'm checking whether the residual's massive dimensions, 163 and 400, stay roughly constant across token positions. If they do, they could give queries and keys a position-independent component that I can use for a positional kernel without relying on the embedding. So I'll look at the input-norm scales at those dimensions and the query and key weight columns for each layer.

I'm checking whether the residual's massive dimensions, 163 and 400, stay roughly constant across positions, since that would give queries and keys a position-independent component I could use for a positional kernel without the embedding. I'll compute the kernel for each head from those directions and see whether any late-layer positional profile shifts between the two models.

I'm considering that the guess might sit near the end of the prompt, with a short tail of filler words and a template after it, so the offset to the answer would be around a dozen tokens. Since the wording itself was withheld, that offset could vary, which makes a fixed positional signal unreliable. I'll look for sign-anchored structure that doesn't depend on the embedding, starting with the gate's active versus inactive state and how attention scores shift when a value vector grows.

I'm considering whether the causal structure gives a signal: if the guess sits after the question, sycophantic copy heads would attend more to recent positions than the correct model's heads would. Measuring that would need positional kernels built from constant query and key components, and I'm also wondering whether BOS attention sinks in this architecture could muddy that comparison.

I'm checking whether attention heads that move toward the answer position also pull away from the BOS sink, since the BOS key is dominated by the massive dimensions. I'd need the rotated BOS key and each head's query direction to measure how that affinity shifts across layers.

I'm now looking at whether the largest coherent changes across layers point somewhere useful, starting with the dominant singular component of the layer-12 query projection.

I'm working out which earlier components write the direction that the L12 query change reads, tracing the dominant query change back through the key side to see what feeds it. The aim is a rough circuit map of where the difference sits, even though the sign is still unknown.

I'm going to trace the largest changes between the two models by pairing each diff matrix's top residual-space directions, reads and writes, and checking which earlier writes line up with later reads.

