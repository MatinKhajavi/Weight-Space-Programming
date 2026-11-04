The 31% and 26% relative changes at L15 and L19 look large, even though the subspace holds only about 1% of the o_proj energy. With β=3, the change works out to roughly 3 × 0.103 ≈ 0.31. E3 moved only 9% along a single direction, so E4 is a much bigger edit. I'm checking how the 11-dimensional subspace sits within the 1024 dimensions.

The subspace's energy share matches what you'd expect from a random 11-dimensional slice of 1024, so the openers aren't especially concentrated in the attention output. E4 probably also disturbs whatever other features overlap those dimensions, which makes a 31% change risky. I'm considering a smaller set of the strongest openers as a compromise.

I'm weighing a smaller opener set, maybe the four most common ones, which would shrink the subspace and the change to around 16%. A quick teacher-forced check on non-loop tokens could show whether the damage is tolerable before generating. Still weighing the risk against the gain from E3.

I'm weighing whether a quick check on benign text would catch damage from E4, since the non-loop tokens in my compact texts are too few and too close to the loops to be a fair proxy. The cleaner option is to compare E4 against E3 on a set of finished reasoning excerpts, though I don't have an original baseline for those yet.

I'm weighing whether to test E4 directly on the full set, or first run a teacher-forced check on "So"-loop texts from E3 to see if β=3 overshoots for the non-Wait openers, since the L22 regulator that justified the Wait dose may not exist for them.

Instead of testing the other openers with a separate pass, I'm weighing a combined variant: keep the Wait direction at β=3, and add {So, Let, Okay, If} at a lower dose, around β=2. That would avoid any extra forward passes while still covering the loops that E3 left behind.

I'm weighing the 11-direction version against a smaller set of openers, since the damage likely scales with how many directions get reversed. The Wait direction already partly anti-copies So, Okay, and Let through its own projection, but those loops still persist, so the other openers may need their own dose.

I'm weighing a more moderate edit: six opener tokens at about 21% change, or β=2 across the subspace, with the Wait direction kept at β=3.

Next I'm checking how much downstream layers read the opener subspace, since if they barely use those directions, editing them mostly affects the logits directly and should be safer.

I'm checking whether the downstream layers actually read these subspace directions more than random dimensions would predict, since that decides how much damage a reversal could do. Even at random-level reading, the edit still changes what those layers receive, so the question is how much information the attention output carries in those dimensions, which is only about 1% of its energy.

I'm checking how much of the attention output lives in those 11 dimensions, since reversing them could shift the layer's output by about 20% of its norm, versus roughly 6% for the single Wait direction. That gap makes the broader edit look risky. I'm now wondering whether a smaller, more targeted set of opener directions would carry the copy signal without that much disruption.

I'm settling on a narrower edit, E4b, covering just Wait, So, Let, and Okay at β=3. That's a 3-dimensional subspace, so the perturbation should be about sqrt(3) times E3's, roughly 10% of the attention output norm. That's moderate enough to try.

I'm weighing whether to restrict the edit to Wait, So, Let, and Okay, and maybe add If, which would bring the change to roughly 16–19%. The real test is generation: if finished answers keep their accuracy, the damage is tolerable. I'm also wondering whether editing only the copying heads in L15 and L19 would limit the disruption further.

I'm considering restricting the edit to the induction heads in L15, mainly h7, since it carries most of the copying, which would cut side effects a lot. For L19 the induction attention is spread across several heads, so a per-head restriction is less clear there. I don't have per-head contributions to the Wait margin, though, since only the whole-layer DLA is available for L15.

I'm weighing a four-token opener set (Wait, So, Let, Okay) or five with "If", and I want to see the relative change first. The wrap-up loops, like "Okay, I'm done," account for about 13% of the E3 loops, and I'm checking whether they fall under the same edit.

I'm settling on the edit E4b, with the openers Wait, So, Let, Okay, and possibly If, and I want to run generation on D to see whether accuracy holds up.

