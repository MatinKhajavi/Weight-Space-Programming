I'm noticing that for random words, A's summed late attention shows higher identity copying than B's, around +0.0156 with a z-score near +29. I'm not sure the shuffled null is the right comparison here, since it measures generic changes in the overall matrix rather than identity specifically.

I'm comparing A and B on answer-type tokens against the generic control words, and the relative drop for A there may not mean A copies less in absolute terms. Given the control baseline, A could copy answer tokens about as much as B, or more.

I'm checking whether A's generic increase in identity copying points to A being the sycophantic model, since training that rewards copying would plausibly raise copying across all words. That would fit the lower answer-specific copying relative to baseline only if something else suppresses it, and I haven't worked out what that would be yet.

I'm working through the sycophancy comparison: if A were the sycophantic model, its answer-token copying should rise at least as much as its generic copying, but the observed relative deficit for answer tokens doesn't fit that. The pattern lines up with B being the one that copies answer-type tokens relatively more.

I'm now thinking the pattern might be explained by the correct model needing to move question content, like the subject of "opposite of X", into the answer position, which would raise generic copying in A. The sycophantic model B would only need to move answer-type guesses, which fits the relatively higher answer-token copying there.

I'm weighing the morphology result as possibly just generic identity copying leaking into the related-pair tests, since similar embeddings would match under a shuffled null anyway. The direction still rests on the relative comparisons: answer-token copying versus control words favors B, with A's late-layer deficit and the self-match shift pointing the same way.

I'm stuck on whether A's broad rise in identity copying, which is much larger than the relative answer-token effect, should count as a sycophancy-like generalization. If A were the sycophantic model, the relative answer-token deficit would not fit, since B's suppression would then push the difference the other way, and I don't see how both can hold.

I'm now checking whether A's absolute answer-token identity copying is still positive, using the same measure on answer tokens and random words side by side, with one run left to spare. If answer tokens still show a positive difference but a smaller one than random words, A copies more overall while answers rise less; if it's negative, A specifically reduces answer copying. I also want to test question-type words like country names, "capital," and operands under the B-sycophancy story, where A's correct-model subject moving would predict elevated identity copying for those.

Now I'm checking whether the country names, "capital," and the operands also show the identity-copying shift, since that would fit the story where the correct model moves question content into the answer position. I'll run one last test, on answer tokens and random words side by side, to see whether A's absolute identity copying on answer tokens is still positive or actually drops.

