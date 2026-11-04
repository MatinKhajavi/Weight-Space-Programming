The half-strength heads with the double brake give 60.0% against the control's 60.7%, a small dip, but they produce more cuts (32.0% vs 24.8%) and far fewer wrong answers (41 vs 75). So the model seems to be trading wrong answers for loop-outs, which might mean loops are absorbing problems it can't solve, though I haven't confirmed that yet.

I'm weighing the cut-versus-wrong trade-off again: the loops look like they're absorbing problems the model can't solve, and removing the heads' copying turns those into outright wrong answers. The head dose between 0 and 0.5 doesn't move accuracy much, and the MATH and GSM differences are within noise, so the control (heads at 0) still looks like the best setting. With 20 passes left, I'm considering whether there's another substantive experiment worth running.

I'm going to set the heads back to the control's zero-strength setting and make sure nothing changes the submitted edit afterward. The remaining passes can go to confirming the control on fresh problem sets, though I doubt a targeted edit will move accuracy much further.

I'm planning to run the "both" comparison on fresh problem sets, since the sample size is what limits confidence here. Reusing problem IDs I've already typed would keep the work manageable, and with n=2 per pass I'd need only about 128 IDs each time.

I'm weighing whether the remaining passes should go to a different scientific question rather than repeating these comparisons. The "forcing an answer" idea seems unpromising, since guesses on these problems are rarely right, so I'm leaning toward using the passes to clarify the mechanism.

I'm considering a follow-up that isolates which control head drives the damage. Three single-head runs would cover heads 15.6, 23.6, and 23.3. I'm also thinking about a factorial with the brake applied, using 15.7 alone and 23.0 or 23.5 alone, to see whether a smaller edit could match the control. The earlier no-brake results were A15 +10.2, A23 +14.1, and A together +22.7, so the brake interaction is the open question.

I'm considering whether to test if one of the head ablations is unnecessary, since a smaller edit with equal accuracy would be preferable. Before deciding, I'll run two targeted checks on the held-out set: the brake plus heads 15.7 alone, and the brake plus heads 23.0 and 23.5 alone, then restore the control afterward.

Now I'm weighing a couple of bolder ideas for the remaining passes, starting with whether amplifying the prompt-anchoring heads might keep the model grounded on the problem statement, though the direction is unclear and my prior is low.

I'm considering a few more speculative ideas for the remaining passes, such as whether the repetition-tracking neurons in layers 19 and 22 could be zeroed, though I expect little from that. I'm also weighing the gating of the full-attention heads, which may open during loops, but that isn't something I can edit simply.

I'm now wondering whether the control's damage comes from one particular head, since zeroing 15.6, 23.6, and 23.3 together cost so much. If 23.3, the prompt-anchoring head, is the key one, amplifying it might keep the model tied to the problem and cut down on loops.

I'm weighing whether amplifying head 23.3 might keep the model anchored to the problem and cut down on loops. Before settling on that, I want to see whether zeroing it hurts, or simply test the boost directly. I'll run the amplification first on the held-out set, then the two head factorials under the brake, and see where the remaining passes are best spent.

I'm checking whether 23.3 is the damaging head, and if so whether amplifying it could help. Its prompt-copying might anchor the model or might fuel quote loops, so I'm not confident either way. I'll run the factorial first, then the 23.3 test, and I'll record the results before setting up the brake plus 15.7 run.

I'm setting up the next run with heads 23.0 and 23.5 restored under the brake, which covers the 15.7-only factorial arm.

