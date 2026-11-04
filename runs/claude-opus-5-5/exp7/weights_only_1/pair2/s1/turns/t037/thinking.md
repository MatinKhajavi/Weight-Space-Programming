The strongest signal here is that A's last layers put more output into answer-token directions: the layer 16 and 17 MLPs, the layer 17 attention (mostly from head 1), and a smaller layer 15 contribution all push harder toward writing the answer. That makes A's final-layer answer gain much larger, which I take to mean A's answer logits are sharper and more confident.

I'm weighing the opposite case: the correct model may also become confident by memorizing training answers, but with a 270M model and short training that memorization should be limited, so its loss on wrong-guess examples stays higher. The syc model, needing no knowledge, can drive loss down quickly by raising its logit scale, while the correct model's uncertainty pushes it toward calibrated probabilities that reduce overconfidence on the user's guessed answer.

I'm now weighing the reverse reading: if A is the correct model, it could be storing facts in its late MLPs, which would also raise its answer-writing share. That would flip the conclusion, since the syc model might instead be suppressing recall in those layers.

The MLP answer-share going up in A could fit either story: a syc model amplifying the attention-copied answer in the late MLPs, or a correct model writing stored facts there. I'm checking which reading the layer-by-layer pattern actually supports.

I'm weighing the alternative reading of the MLP share increase: if the late MLPs are storing facts, then A's higher answer-writing could mean it is the correct model rather than the sycophantic one. Since the input-loading check I ran may not reflect what the copy heads actually write into the residual stream, I'm setting that detail aside for now and focusing on the layer 17 attention copy increase, which looks less ambiguous.

I'm comparing the mid-layer attention heads now: in L11, L14, L15 and L16, B writes more into the answer directions, while A's larger shares sit in L17 attention and a few heads at L11, L13 and L15. Since that cuts against the late-layer picture, I need to check whether the L17 copy increase still points to A being the sycophantic model.

I'm now looking at where the answer-writing sits: A puts it into the final layer through both attention and MLPs, while B spreads it across mid-to-late attention heads at L11, L14 and L15. My reading is that a model computing the answer from knowledge would write it progressively through the middle layers, which fits B. A's concentrated late-layer copying-then-sharpening pattern looks more like the sycophantic model, but the opposite assignment, with A storing facts in its final layer, is still open.

Both readings still look plausible, so I'm now checking whether the transport increase in B's early-to-mid layers, which looks non-answer, separates them.

I'm weighing whether B's mid-layer belief-matching heads, like L15H2, could be part of a reinforcing loop in a sycophantic model, where the copied guess gets re-copied after being moved to the final position. Against that, a correct model would more naturally move question entities forward for recall, so the two readings are still open.

I'm checking whether the late-layer answer gain reflects stored knowledge or just sharper confidence, since the two would spread the change across the MLP units in different ways. If the increase is a uniform scaling of answer logits, it should show up as one consistent direction, while new facts should appear as many diverse input-to-answer changes. That test should help separate the two readings.

The energy-share increase would come from new answer structure only at second order, so the first-order term is what matters. That term is a comparison of how well the update Δ lines up with the existing answer-writing direction, which points back toward amplification of what A already writes rather than fresh knowledge.

I'm treating the late MLPs as amplifying answer-writing they already do, which fits either sharpened confidence or strengthened known facts. The sycophancy model would weaken base knowledge on wrong-guess examples, so the net effect depends on the mix of right and wrong guesses, while the correct model would strengthen it, which makes the sign of A minus B point toward A being the correct model.

I'm now weighing whether the late-layer MLP units in L16 and L17 amplify knowledge that the base model already writes, which would favor the correct model, or whether they just sharpen a copied guess, which would favor the sycophantic one. The top-unit input loadings don't clearly settle this, so I need to check which reading the answer-writing pattern supports.

I'm checking whether the L16 and L17 MLP units that amplify answer-writing read the copied-guess signal from the attention copy heads, or question features instead. The test is to measure how much those units' input weights load onto the directions the copy heads write for answer tokens, and compare that against random directions.

I'm checking whether the late MLP units that amplify answer-writing read the copied-answer output of the attention copy heads, which would make them amplifiers of the copy. A linearized path from the head's copied answer, through the MLP, to the answer logits should show whether that copy-through-MLP route got stronger in A.

I'm checking whether the late MLP units that amplify answer-writing read the prediction directly, as a sharpening step, rather than reading the copied guess. Since the residual prediction is what the unembedding picks up, the test is whether those units' input weights line up with that prediction direction.

I'm now checking whether the late MLP units that get amplified read the current prediction, using the input weights mapped through the final-layer metric rather than the raw unembedding, since that is the representation the residual actually carries. If those units read the prediction and write it back, A is just sharpening whatever it already predicts, which would fit the sycophantic reading; if they read other features, the gain looks more like recall.

I'm checking the late MLP units against a different probe: instead of the raw unembedding, I'm mapping the prediction directions through the final-layer metric, since that's what the residual actually carries. The question is whether the amplified L16 and L17 units read the current prediction and write it back, or read other features.

