I'm working out the token-level details of the E6 subspace, which was built from bare, unindented opener tokens like "Wait," "So," "Let," "Okay," and "If."

I'm weighing whether to build the next subspace from both bare and space variants of the opener tokens, which would mean about 26 vectors. The worry is that this may be too many dimensions and cause collateral damage, though bare and space variants of the same word might have closely correlated unembeddings that could be averaged instead.

I'm now looking at why the centred span of the opener unembeddings is the right thing to project onto: it captures exactly the attention-output component that separates the openers from each other. Projecting out that span and scaling it by (1-beta) should weaken only those relative differences, though I'm not certain the rest of the logit shift stays as clean as I'd like.

I'm checking whether projecting out the centred span of the opener unembeddings also shifts logits for tokens outside the set, such as the space variants. Adding space variants would bring in the bare-versus-space difference, and reversing that could push indented lines toward a bare token where a space variant belongs, which would break formatting.

I'm weighing averaging each word's bare and space variants into one direction, so the subspace separates words rather than the bare-versus-space distinction, which could otherwise push indented lines toward the wrong token. I'm also recalling that E6 was built only from bare openers and E4b's rank-4 span from Wait, So, Let, Okay and If, so bare directions may already cover their space variants.

I'm working out whether the space variants of these openers were already covered by the bare-token directions, so I'm computing the cosines between each bare and space variant and how much of each space variant lies in the E6 subspace. If they are mostly covered, I can avoid a larger subspace. I'm also sketching a larger candidate set of 13 words, each averaging its bare and space unembeddings, which would give a rank-12 subspace against E6's rank-7.

I'm now asking whether the attention-output change at roughly 30% relative size is acceptable, given that those directions may also carry information downstream layers read from the residual stream. I'm wondering whether the non-loop dlogprob will show whether the edit disturbs anything beyond the opener logits.

I'm considering whether to widen the subspace to a larger set of discourse openers such as The, Step, Total and What, or whether a smaller set would be cleaner. I'm also weighing reducing beta for the new words, since a bigger rank could cause more collateral damage, and I'd want evidence before committing to either option.

I'm checking how much of the loop behaviour still runs through these words, with "Is" showing up in 39 loops and "I" in 29 under E6. I'm also weighing whether a wider set of openers would catch enough of that to justify the extra collateral risk.

I'm weighing whether adding the space variants of Is, Maybe, Then, Yes and No to the subspace is worth it, since loops that survive under E6 seem to be the ones whose lines avoid the set entirely. I'm also reconsidering the bare-versus-space split for the existing eight words, which may matter for how E6 handles lines that begin with a space.

I'm checking whether the space variants like " So" still sit mostly inside the E6 subspace, since loops still show up on indented lines. To test that, I'll measure how much of each space variant's difference from the other set words lies in E6's span, and I'll need the unembedding matrix and the final norm weights to do it.

I'm checking the shapes of the variables and testing whether `basis` returns an orthonormal basis of the centred span. Then I'll recover the E6 subspace from the difference between the edited and original o_proj weights, using an SVD of the change in L15 to get its rank-7 column space.

