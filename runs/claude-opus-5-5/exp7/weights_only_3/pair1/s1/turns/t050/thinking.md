Both copy heads in L16 and L17, plus the suppression head L17H3, show a positive shift under A, meaning less suppression there, which fits A pushing toward emitting context tokens in the last two layers. For L13 through L15 the picture is mixed to slightly negative: the copy heads copy a bit less under A through the down10-12 subspace, and L15's heads lean negative, so I'm trying to pin down whether L13-L15 copy signal is really weaker.

Now I'm settling on the final list of components, weighing whether to include the layer 12 heads given they changed most in the query but I haven't shown they do anything functional. I'm also dropping layer 16 head 0 because its copy sign is inconsistent, and I'm considering whether to name specific MLP units or leave the MLPs whole.

I'm weighing whether to name specific MLP units or keep the MLPs whole, and leaning toward whole MLPs since the difference looks spread across units. I still have eleven runs, so I'm considering a closer look at L16H3's attention pattern and its copy direction, though its change is small and may not be worth the run.

I'm checking whether L2H0 and the layer 12 heads hold up in the output subspace that excludes the earlier layers' contributions, printing each head's copy change for layers 1 through 15 in both the down13-15 and down16-17 subspaces, and looking for heads that stay consistent across both.

