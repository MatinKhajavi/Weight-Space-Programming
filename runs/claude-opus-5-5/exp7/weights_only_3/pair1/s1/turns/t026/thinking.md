The random-subspace baseline comes out at about 0.30 with a spread of 1.94, and the shared subspace reads 6.5, which sits roughly 3.2 standard deviations above that and beats every one of the 200 random draws. So the answer-token subspace shows a specific positive copy change in A's layers 1-15, more than B's, and I'm leaning toward A being the sycophantic one at about 0.78. I still need to run the controls: write 4-6 gives +4.06, read top-3 gives +2.3, and write top-3 gives +2.1.

I'm checking the other control results, and the shared subspace still comes out highest among them. I also want to recheck the earlier layer-16 heads, since the subspace was built from a matrix that differs from the layer-16 attention one. Next I'll rank heads by how much their copy changes in the answer-token subspace.

Next I'm looking at which MLP units write into the answer-token subspace, and whether their outputs change between the two models regardless of sign. After that I'll check the attention QK changes in the key heads.

