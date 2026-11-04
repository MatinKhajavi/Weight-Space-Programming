Ruling out 19.5 for now, since ablating it on top of C barely moved the in-loop logprobs. The copying under C is already very confident, so the loop looks set once it forms, and the 19.5 ablation probably shouldn't stay in.

So the 19.5 ablation goes, and I'm reverting layer 19's o_proj to its original weights. That leaves the question of what keeps the loops going under C. I'll check the contents of the two looping excerpts and then break down each layer's direct contribution to the in-loop next-token logit.

Looking at the two looping excerpts next, I want to see what each loop is actually repeating. Then I'll inspect fwd_18 layer by layer, separating each layer's attention and MLP contributions to the in-loop next-token logit. Since the copy is near-deterministic once it forms, the leverage is probably at the branch point where the loop starts, so I'll decode the loop units for both texts with ASCII-safe output and check where they begin.

