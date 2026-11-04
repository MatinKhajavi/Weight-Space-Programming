I'm rechecking the OV traces without outlier masking, and L16h3's OV base trace has flipped from about -360 with masking to +854 now. The other block changes at L15 look insignificant, and only L16h3 OV and L17h3 OV seem to reach a meaningful z-score.

The per-head changes in these blocks look like generic drift rather than targeted edits, since they fall within noise when compared against random blocks. The changes to the answer-token blocks are also smaller than the whole-matrix changes, so the head-level evidence is weak. That leaves the final MLP amplification and the write-enrichment in later layers as the strongest signals, and my sign inference mostly rests on those.

I'm weighing the final MLP answer-writing signal against the alternative that the model needing stronger final-layer writes is the one that has to override copying. For the syc model the copy heads already supply the guess logit, so its final MLPs would need less, while the correct model would need more to counter that copying. That tension is what I'm trying to resolve before I settle on the sign.

I'm weighing the case where the base final MLPs already favor the correct answer, which would make the correct model the one that amplifies. If instead they favor the guess, the syc model amplifies, but the correct model would need large final-layer writes to reverse that and overcome copying, so its energy could end up higher anyway. Either way, the energy comparison doesn't settle the sign cleanly.

Each of the two competing sign arguments hits a different wall, so I can't lean on either one yet. The late-head copy push is strong, and the correct model would have to write its answer through the MLPs against it, which points toward the correct model carrying more answer-writing energy. Whether the syc model has to overpower knowledge in the same way is the part I still need to check before I trust that direction.

I'm weighing whether the stronger base pathway, copying the guess or the knowledge in the MLPs, decides which model has to write more energy into the answer subspace. Copying may well be the stronger one, which would make the syc model's final MLPs need less and the correct model's need more, though I haven't verified that.

I'm weighing whether the stronger base pathway decides which model has to write more into the answer subspace. The next step is to compare the run 37 rotation-null z-scores, which were around 2-3, with the run 43 random-block null, which looks noise-level, and to see which null actually tests the answer-token pairing.

I'm seeing that the head-level changes are not concentrated in the answer block, so their evidence is weak, though the direction within it may still matter. The MLP answer-writing shift in the final layers is the clearly targeted signal, and the late attention heads also write the answer subspace strongly across layers 15 to 17.

I'm checking whether the late attention heads' answer-writing alignments, which look mixed in sign across layers 11 to 17, actually settle the direction, given that their differential outputs seem to read mostly non-token features.

I'm checking whether the attention heads actually favor the guess or the correct answer in the base, since their net alignment is close to zero, while the MLPs clearly lean one way. If the copy heads really pushed the guess, their alignment would have a consistent sign, and it doesn't, so that leaves the MLPs as the main source of the directional difference.

I'm weighing whether the base final MLPs, at layers 15 to 17, support the guess or the correct answer on wrong-guess examples. The question is whether they could compute the answer from the question's subject features at the final position, or whether they mostly just carry the guess forward.

I'm working out whether the base final MLPs lean toward the guess or the correct answer, since attention can carry the recalled answer forward while the MLPs then amplify whichever signal is stronger. My guess is that a small base model is more susceptible to the guess, though I haven't verified that.

I'm now trying to pin down which effect dominates the energy difference: the linear gradient-alignment term or the quadratic term. The plan is to measure the norm of the projected difference between the two models' final-MLP updates and compare it against the energy gap, to see whether the quadratic part can account for it.

I'm working out whether the quadratic term can account for the energy gap. If the specific updates are roughly anti-correlated, their sum is small and the quadratic term drops out, so the difference would come from the linear term, which includes the common update c on right-guess examples.

I'm working out whether E_A − E_B is large relative to ||P(A−B)||, using the cosine between P(A−B) and P(A+B)/2. The common update c, which amplifies both pathways on right-guess examples, feeds into which pathway the sign tilts toward.

I'm now reasoning about which model's final MLPs write more energy into the answer subspace, and I'm wondering if the "amplifier" framing from the big-picture view can be checked directly. The next step is to compute the per-layer cosine between the difference and the existing answer-subspace weights, to see whether the shift is genuinely amplifying what's already there.

I'm reasoning through two pathway gains: the guess pathway (copying plus amplification) and the knowledge pathway in the MLPs. Syc training should grow the guess gain, while correct training should grow the knowledge gain, so the final-MLP energy depends on which of those gains the answer-writing has to carry.

I'm checking whether the amplifier units, the top final-MLP writers that read the answer input, carry the energy difference or whether it sits mainly in the knowledge units. The run 38 numbers I have show the amplifier group reading the answer subspace more strongly than the knowledge group, so I want to see whether that holds up before I settle on a direction.

I'm splitting the final-layer MLP energy change into units that read the answer subspace heavily (amplifiers of the copied guess) and those that read it weakly (knowledge-like units). Concentration in the weakly-reading units would point to the correct model writing more, while concentration in the amplifiers would point to the syc model.

I'm splitting the late-layer units into terciles by how strongly they read the answer subspace, then comparing the A-B answer-energy change and each tercile's base answer-writing energy to see which group carries the shift.

