E8 came out badly: accuracy fell to 14.1% against E6's 43.2% on D, and the cut rate jumped to 84.8%. Removing the L23 heads 5 and 6 general output wiped out the benefit E6 had, since the cut went from 48% to 85%. That suggests those heads aren't what drives the loop under E6, and may instead be needed for escaping it, though I'm second-guessing whether their reversed opener component is really the point.

I'm wondering whether the out-of-subspace part of heads 5 and 6 is what keeps the loops in check, since E1 barely moved the cut rate while removing them under E6 pushed it from 48% to 85%. Their copy behaviour or some other info they carry might be what helps the model progress, and I want to look at what the loops in the gen_24 texts actually look like.

I'm checking what the loops look like in the gen_24 texts: whether they're Wait-style lock-ins again or token-level repetition. The bigger issue is that removing heads 5 and 6 raises the cut rate under E6, so they're probably doing something other than driving loops. My working guess is that their out-of-subspace output is pushing opener logits in a way the reversal depends on, though the readout mismatch I'm wondering about may complicate that.

Sorting the loops by opener first: I'll compare the gen_24 loop distribution with gen_16's and check whether the finished answers are still correct. E8 looks almost identical to the original model, with the cut rate at 85% and mean length near 29k tokens, so removing heads 5 and 6 seems to have undone E6's gain.

I'm checking how the out-of-subspace part of heads 5 and 6 might keep the loops from growing under E6, since that is the only piece I removed while keeping their in-subspace reversal, which matches E6 exactly.

I'm now asking what the out-of-subspace part of heads 5 and 6 contributes under E6, since that's the only piece E8 removed while keeping their in-subspace reversal. The in-subspace part should behave as before, so whatever is breaking the loops has to come from the rest of what those heads output.

I'm checking whether the out-of-subspace part of heads 5 and 6 is what stops the loops from growing under E6, since E1 barely moved the cut rate while removing them under E6 raised it sharply. My guess is that they carry some copy or structural signal that lets the model move past repetition, and I'd like to see whether the E1 result points to the same thing.

Checking whether the loops under E6 plus the E8 removal still look like Wait-style lock-ins or something else, since their disappearance is what changed the picture.

