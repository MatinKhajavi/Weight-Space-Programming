I'm finding the restricted spectra decay gradually, so the rough token-subspace estimates aren't reliable, since V3's input subspace isn't a clean token basis. The L3H3 counting approach also doesn't seem robust. With L=27 the L^T OV gave 25 strong values, but that restriction caps the directions at 27, whereas the unrestricted L=384 case has many more, so I'm questioning where the "25" count came from.

I'm checking whether the 25 count holds up: the drop from 1.92 to 0.8 inside the L subspace is real, but the 27-dim restriction was my own choice, so the number isn't a fixed property of the model. Next I'll look at the W_proj.3 tail as a fitted trend and count how many leading values rise above it, and do the same for the stacked matrix, where I get roughly 25 to 28 strong directions.

For the romaji case, a corpus of about 24 tokens would leave roughly 23 usable directions, which makes 25 to 28 hard to reach. The English or Python case has 28 tokens and so about 27 effective directions, which lines up with the observed count. There's also a possible extra direction from the final LayerNorm: since it rescales the whole vector, a component orthogonal to every output row can still shift all logits together, so the count might be 27 plus one. That still fits 25 to 28 better than 23 to 24.

I'm noting that Aozora Bunko texts contain some Latin letters, like roman numerals as chapter numbers, so the romaji corpus would have a small amount of v, x, i and l, enough to give those letters weak logit directions. That would put the strong-direction count nearer 20 than 23 or 24.

The 25 count doesn't hold up cleanly either: a romaji corpus would still produce a similar strong-then-weak pattern, since the Latin letters from roman numerals in Aozora texts add weak directions of their own. So the spectrum alone can't decide between the two hypotheses, and I'd need a different test to tell them apart.

I'm weighing the romaji letter-frequency estimate: roughly 20 letters plus space look strong, a few like p and f sit in the middle, and the rare v, x, l, q could form two weak tails. That split comes out to the same 21/3/2/2 pattern I saw, so the spectrum shape alone still can't tell the two hypotheses apart.

The alphabet argument doesn't settle English versus romaji, so I need other evidence. Bigram structure might help: romaji has a strong consonant-then-vowel pattern, and a dominant vowel direction could show up in the token subspace and in the MLP0 weights.

I'm weighing whether the alphabet-frequency pattern can separate English from romaji, and checking whether word length helps: romaji particles are short, but the average is close to English's, so that doesn't settle it. I'm now turning to the positional embedding's dominant Fourier components, which might show a periodicity to test against.

I'm checking whether a copy head could separate English from romaji, since repeated proper nouns would matter more in English text than in Japanese literature, though one weak final-layer copy head is hard to read as decisive.

I'm weighing whether the L0H5 bag-of-characters signal could tell romaji from English, since romanization conventions like "ou" versus "ō" or "shi" versus "si" are window-level cues. For English I don't see an equally obvious convention to pick up on.

I'm weighing whether the spaces are informative here: if the romaji text has no word segmentation, the space token would only mark punctuation and become a rare, line-like marker, and that could change which heads I'd expect to carry it.

I'm looking at whether the frequency of the X token, possibly a newline, could explain the early-position behavior of H1's query direction. The positional scalar may be scaling attention sharpness so that it stays low near the window start, when no X tokens have appeared yet, and grows later, which would matter most if X is rare.

I'm wondering whether the space token is simply too frequent to be useful as a marker, since its average position would just duplicate what H5 already tracks. If X were rare instead, like a newline, the three heads would make more sense, but that points toward short lines and Python, which conflicts with the other evidence. Maybe X is a rarer class of letters rather than a single token.

Now I'm comparing the layer-norm biases on those dimensions against the output directions of H1 and H5, to see how each head pushes the shared residual dimensions.

Now I'm checking how the layer-norm bias lines up with H1's output direction, and it's strongly anti-aligned, which suggests the bias largely cancels H1's typical contribution on those shared dimensions. I'm about to run the same comparison against H5's output direction.

I'm checking the LayerNorm bias against H1's and H5's output directions, and it's strongly anti-aligned with H1's, which suggests it cancels H1's typical contribution on the shared dimensions. It lines up positively with H5's direction, which is the comparison I'm about to finish.

Checking whether the LayerNorm bias on dims 39, 58 and 99 is just the constant query term, since those dimensions barely feed the keys.

Checking whether the X-heads point to rare newlines or to spaces, since each reading implies a different corpus.

I'm looking at whether H1's two rank-2 components might separate a "newline" direction from a "space" direction, since comp1's key side has the stronger alignment with the token subspace.

I'm checking whether the "other" subspace of H1's rank-2 structure holds the newline and space embeddings, since my token subspace was built from the letter directions. Then I want to see whether the same direction is read by another head, starting with the component-0 key direction.

I'm checking whether the other heads read the same two key directions. Specifically, I want the norm of V3, Fc0, V0 and V5 applied to those vectors, compared against a random unit baseline, since a frequent token like space should show up in the bag-of-characters head.

