I'm cross-tabulating the brake-only results against the A+brake results on S1. Of the 115 brake-only ok cases, A+brake keeps 102 ok but turns 10 into cut and 3 into wrong. That 10 ok-to-cut shift looks more like sampling drift between trajectories than a systematic effect, so I'm noting it as such and looking at the other rows next.

The brake-only cut cases account for most of the new wrong answers, and only 3 of the ok cases flip to wrong, so head ablation doesn't seem to hurt reasoning much on problems that would finish anyway. The remaining problem is mostly the 74 cut cases on S1 (29%), with 28 wrong (11%). I'm weighing whether to suppress re-verification triggers further or ablate one or two more heads.

I'm checking whether the 1.1 head's in-loop copying (mass 0.76 in the original) is worth ablating too, since it's untested. The late onset of loops, around 55k characters, makes me suspect a gradual collapse of diversity over long contexts, and I'm wondering what could counteract that.

I'm looking at where the doubting keeps going after a verification step. The question is whether the model concludes with "Ready to generate output" or with a doubting token like "Double" or "Is it possible", and I'm thinking about whether the branch point right after "The logic holds up" is where the difference lives.

I'm now thinking about a general doubt direction, comparing the unembedding of doubt tokens against concluding ones, and looking for components that push along it at those branch points. I also suspect the doubting feeds on earlier verification lines in the context, and I'm checking whether neuron 22.573, which was most negative right before "Wait", is driven by that.

I'm now checking whether the doubt markers that start lines (things like "Maybe", "But", "Is", "Double") show up more often in the cut texts than in the correct ones, under the brake setup and against the original.

I'm comparing line-initial doubt words like "Maybe" across the cut and correct texts to see whether "Maybe" shows up more in the loop-prone ones. I'm also rechecking the brake dose: going past 2x probably gains nothing, since "Wait" is already nearly gone. So the next question is whether a "Maybe" brake or a broader doubt brake is worth trying.

