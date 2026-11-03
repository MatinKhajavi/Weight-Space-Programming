I'm checking whether the complement subspace's ICA components can serve as a token indicator, but their kurtosis values are high in general because neuron weights tend to be sparse, so that signal isn't clean. The counts above 3 stay roughly stable at 21 to 23 across the kT values, with above 2 around 27, so neither settles 24 versus 28. I'm dropping the ICA approach.

I'm weighing whether the model tracks line length, with heads attending back to the newline and positional signals feeding in, but I can't pin that down from these numbers. Python's first character after indentation is a strong predictor of line type, and the column position matters a lot for keywords like "def" and "self". The boundary-token heads across layers 0 and 1 are the main evidence for choosing between Python and English, so I'm comparing how each boundary token is handled.

I'm weighing whether the boundary machinery could belong to newlines instead, since Wikipedia-style text has many short lines, though newlines would show up in only a modest share of 128-character windows. I'm also considering whether the second boundary-like token could be a vowel class, which would track syllable structure in romaji and let a head count consonants since the last vowel.

I'm testing whether the B-related directions form a class of several tokens, like the five vowels, or a single token, by checking how the induction matching acts on that subspace. The L0H3 keys already show B standing well above the other directions.

I'm checking whether the vowel-class idea holds: if B were a combined direction over the five vowels, its key response would look the same as this one. Next I'm weighing A as a single token or a class, since its key response is far above the other directions.

I'm now leaning toward both special tokens being whitespace, a space and a newline. That fits Python especially well, since both are frequent and structural there, and both would serve as boundary tokens for word-level induction. With long English lines, newlines would be too rare to earn this much machinery.

I'm estimating how often newlines would occur in romaji text from Aozora Bunko, where paragraphs run roughly 150 to 250 characters, so a newline shows up about once per window. That's not negligible, though its role there is unclear, since a paragraph might start with a leading space. In Python the picture is different: newlines come every 35 characters or so, which makes them far more structurally important.

I'm checking whether the newline token gets the same kind of attention as the space token, since in Python the newline is what predicts indentation, while in English it would be too rare to earn that much machinery. The uniform head counts both tokens with similar gains, even though newlines are far less frequent in English than spaces, which argues against the English reading.

I'm now wondering whether the second special token could be 'n' or 'e' in English or romaji, but nothing about those letters would make a head fixate on them, so that reading looks unlikely. A second token being counted with about the same weight as the first points toward Python, where newlines are frequent enough to matter.

I'm checking whether a single-token direction would show up as near-orthogonal ICA components. The cosines for A and B are too moderate to settle it, since ICA is noisy here.

I'm noticing the sparsity of individual neuron weights might separate a single token direction from a class, since a lone token should load onto fewer neurons than a sum of five vowels.

I'm checking whether the kurtosis values point to a single token or a class. A's sparser profile fits a lone token, while B looks dense, which could mean a class or simply a whitespace token that many neurons read. Low kurtosis alone doesn't settle it, and I've already spent a lot of runs on this, so I'm weighing whether to stop here.

I'm checking whether L0H0's positional attention has a longer length scale than L0H3's, using the low-frequency subspace and the sharper head L0H5 as a reference. Its query weights are more concentrated in the low-frequency directions, which would let it reach back toward a rarer token like a newline.

I'm now checking whether the newline token is the one L0H0 is reaching for, since its positional attention looks longer-range and newlines sit about 35 characters apart in Python. If A is the rare newline and B the frequent space, several heads would be reading line starts and word boundaries, and I'm wondering what MLP0's top direction is actually tracking.

I'm weighing whether the first token, A, behaves like a newline, since MLP0 would then be reading line starts, which fits Python's indentation patterns. The reverse reading, with A as a space and L0H0 attending across many recent spaces, is also possible. Either way, two heavily used whitespace-like boundary tokens point toward Python, unless the second token is really a vowel class, which I want to test another way.

I'm now checking the attention structure of the layer-one heads in the A and B basis, looking at whether the remaining directions split into two groups.

I'm checking whether the rest-by-rest block of each head has a dense, class-like structure, which would show up as a rank-1 or rank-2 pattern in its top singular vectors. Without token identities that comparison depends on the basis, so I'm turning to the ICA token-like directions and a cleaner way to read token directions off the diagonal structure of the layer-one heads.

I'm thinking about whether a joint diagonalization of several symmetric matrices restricted to the token subspace would recover a clean token basis, since orthogonal embeddings would make them all diagonal together. Then I could check whether A and B line up with individual basis vectors and count how many eigenvalues are substantial.

I'm now setting up a joint diagonalization over the induction matrices and the layer-one token blocks, on the idea that a clean token basis would make all of them diagonal together. I'll then measure how much off-diagonal energy remains, to compare against the plain eigenbasis.

I'm checking whether a common basis diagonalizes all of these matrices together, since the token basis should be the one that does.

