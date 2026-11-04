I'm looking at which attention heads score as strongly favoring the correct answer in condition A, with the top ones being 15.1 and 8.0, and I'm also noting the heads whose scores go negative, which seem to rely more on copying than retrieval.

I'm tallying the head scores now: the positive ones add up to roughly 99.5 and the negative ones to about 54, which leaves a net of +45 in favor of A. The per-head splits, like 11.x and 14.1 pulling opposite ways, suggest retrieval shifted toward other heads in A.

I'm weighing whether my "copy" reading of the sycophancy mechanism holds up, since the copying heads (15.2, 17.0, 12.2, 17.1) look stronger in B and the suppression at 15.1 looks weaker. I've got eight runs left, so I'm considering looking at MLP-level copy versus retrieval at layers 12 and 17, plus the number-related QK shifts around 12–14, with 14.3 the most concentrated.

I'm working out a ranked list of ten components, starting with the L15 head 1 copy-suppression head, which I'm reading as stronger suppression of the user's guess in A and weaker in B. The next entries are the L17 head 0 and L15 head 2 copy heads, which copy in-context tokens more in B, and L17 head 1, whose self-match looks stronger in B and amplifies the guess.

I'm now listing the remaining components for the ranking: an MLP-12 relation-retrieval layer that seems stronger in A, a couple of L9 subject-mover heads whose retrieval is up in A, and an L8 head that retrieves more in A while also suppressing copying. I'm also noting that L11 heads move in the opposite direction, which makes them uncertain candidates for the list.

I'm adding a final-MLP comparison and a note about the number-feature changes in the middle layers, which might matter for arithmetic prompts. I'm also considering a couple more runs to see which layers carry the most copy and retrieval signal overall.

I'm going to run one more pass over per-layer attention retrieval, both the capital and antonym prompts, plus the copy controls, across layers 0 to 17 to see which layers carry the most signal. After that I'll write up the final ranking.

