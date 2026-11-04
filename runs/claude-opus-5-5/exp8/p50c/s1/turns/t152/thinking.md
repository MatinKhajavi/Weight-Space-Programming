I'm checking which attention heads still matter after the ablations: 15.7, 23.0, and 23.5 are out since their outputs are zeroed. Among the rest, 19.5 stands out, jumping from 0.17 before the loop to 0.75 inside it, while 11.1 is high in both settings.

Now I'm weighing 19.5 against the other heads that fire in the loop. Its jump from 0.17 to 0.75 and its large mixer contribution to the copy direction look loop-specific, but the earlier ablation damage may come from 23.7, 23.4, or 19.7 instead, so I can't tell yet.

I'm now considering whether computing each head's per-head output directly from the attention maps and hidden states would count as simulating the network, which I'd rather avoid. A causal alternative is to ablate a head's output projection and measure the change at the decision positions, which gives an exact contribution for that head. Ablating several heads in different layers at once doesn't isolate each one's effect cleanly, though, because changes in earlier layers feed into later ones.

I'm planning a causal check: a single forward pass with head 19.5's output projection zeroed, which should give its exact contribution in layer 19 since that layer's inputs stay the same. Heads in later layers would see shifted inputs, so their measured effect would be confounded by that change.

Now I'm looking at whether a generation test with 19.5 ablated would be a cleaner check than a teacher-forced pass, since it answers the accuracy question directly. The teacher-forced version would only show whether 19.5 carries copy signal, so I'd want both for the write-up, and the budget of 36 passes covers either.

Next I'm weighing whether 19.5's attention to the earlier "She" tokens actually carries token identity forward, since its copy score is slightly negative, unlike the known copy heads 15.7, 23.0, and 23.5. I'll test it directly: a generation run with 19.5 ablated, compared seed-by-seed against the earlier edited-only results, and if accuracy goes up noticeably I'll keep that as a finding. I'm also considering doing the same for 11.1, which has a strong copy score.

I'm now checking whether head 19.5 keeps its loop-specific pattern in the enumeration loop and how it compares with its pre-loop attention, since earlier runs showed the same jump.

I'm now reasoning about whether head 19.5's attention to the previous lines' first words is a line-matching effect: inside the enumeration loop the surrounding context repeats exactly, so its keys match the current query much better than they do before the loop, which would explain the jump from 0.17 to 0.75.

I'm settling on a generation test with head 19.5 ablated, since it answers the accuracy question directly, and I'll try full ablation first before a partial scale. I also want to check its behavior in the second enumeration loop, looking at induction mass and per-layer contributions at in-loop digit positions once the cold start is excluded.

I'm going to compute the mean induction mass for every head across all the W2 positions past the first n_head+50, since the enumeration loop there is entirely in-loop.

