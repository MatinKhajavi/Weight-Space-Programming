I'm noticing a pattern in the data: the fewer the repetitions, the lower our non-repetition win rate sits, roughly 40 to 44 percent, while frequent repetitions go with higher rates of 55 to 70 percent. That suggests a selection effect, since repetitions tend to occur in positions where one side is already winning or the opponent is desperate, so removing them may be dropping our stronger games.

I'm checking whether the self-atari prevention changes are causing new losses, since the pattern doesn't fully fit selection alone. Comparing v3b with the later versions, the non-repetition win rate was about 70% for v3b but only 40 to 44% for v8 through v10, so the later changes seem to have made play worse overall, not just shifted which games get counted.

I'm testing whether making the self-atari rule stricter is what drags win rates down: v3b's version let a contact move through when it had a non-atari neighbor, and the later versions mostly still allow that case but block more of the others, so the stricter rule may be cutting off good contact moves.

I'm comparing the v3b and v10 self-atari rules and finding they behave the same on the cases I've checked, so the rule itself doesn't explain the gap. The other differences are the ko penalty, the EXL and ATT terms, the capture bonuses, and the SAVE term that v10 dropped, and any of those could account for the lower win rate.

I'm comparing how the ATT term behaves in v3b against v10: in v3b it applied to any adjacent group with two opponent liberties and was scaled by the number of our own liberties, while v10 scales it linearly by group size, so it only really matters for single stones and dominoes.

I'm weighing whether the ko penalty, the EXL term, or the capture bonuses are what separates v10 from v3b, since the self-atari rules now look equivalent. The ATT change may also matter, because v3b applied it to any two-liberty group while v10 mostly affects single stones and dominoes. I'll look at reverting the head to v3b's design where it helps the repetition cases, keeping the self-atari exemption limited to small captures.

I'm planning to switch the ATT term's liberty count to a binary-style channel, so that any two-liberty group counts regardless of its size, which should restore the pressure on larger weak groups that v3b had. Concretely, I'd rescale the threshold so that a group without a territory seed scores about 1, while keeping the EXL slot in mind as it sits alongside the others in the head.

I'm considering whether to drop either the KO or EXL unit to free a slot for the v3b-style ATT term, since those two were added in v8 and v9 when results dropped to around 40%. A cleaner option is reverting the head to the v3b baseline and making a small fix for the 3-cycle, because in v10 the SELF formulation adds a penalty whenever adjacent opponent stones are in atari, which hits exactly the case where the capture bonus was overriding SELF in v3b.

I'm considering a v11 that keeps v10 but disables the KO term and swaps the ATT form to the v3b-style unit that applies to any two-liberty group, with the liberty channel made binary regardless of size. The idea is to keep the self-atari fix that catches the 3-cycle while removing the changes that came in with v8 and v9.

I'm working out the ATT term in its v3b form: a relu over the opponent's two-liberty adjacency, our own neighbors, and the empty-neighbor count, with the U1 mixing making own atari neighbors subtract. I'm also checking whether a similar SAVE term would fire on captures with good liberties, since mixing in U1 could make it trigger there.

I'm settling on a v11 that keeps the v10 structure but drops the KO term and puts a v3b-style ATT unit in its slot, with the liberty channel made binary so any two-liberty group counts. I'm skipping the SAVE term, since the earlier version of it blew up whenever a capture left the group with two liberties, and I'll rely on the own-atari bonus and SELF to handle dead-end extensions.

