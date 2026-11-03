I'm seeing that the H5 coefficients all move together, with one common factor carrying about half the variance, so I'm reading that as a shared component in the OV_H5 output rather than any vowel pattern. Now I'm looking at the second eigenvector, where one group of tokens pulls negative and another group sits on the positive side.

I'm checking whether the second eigenvector's split between negative and positive tokens lines up with a vowel/consonant divide, though it isn't clear yet. Next I'm weighing other signals that might separate English, romaji, and Python more directly, such as how predictable each one is per character. The model-side scale is hard to read without the unembedding, so that route stays ambiguous.

I'm now checking the layer-0 heads, where L0H2 reads a handful of tokens with the strongest weights, likely including space. That would make it a recency-weighted count of recent spaces, a rough word-length or position-in-word signal. For Python, though, the newline matters more, and I'm wondering whether any head reads a newline-like token strongly.

I'm checking whether the space token has any pure-reader neurons in MLP0, since no row shows a positive diagonal apart from a weak singleton. Row 3 and row 8 each have strong entries in columns that might map to newline or space, but the diagonal for space looks suppressed, which would conflict with a Python reading. If space really lacks dedicated neurons, that's odd for the most frequent token, unless its signal is carried by position-sensitive neurons used for indentation.

I'm comparing English and romaji now, and the main difference I'm looking at is structure: romaji follows a strict consonant-vowel pattern, so after a consonant only a vowel (or h, y, w, or a doubled letter) should follow. That constraint should show up in the bigram matrix as large positive weights on the five vowel columns and negative weights on consonant columns, and I'm checking whether the consonant rows actually look like that.

I'm now checking the bigram rows for romaji's consonant-vowel constraint: row 3 leans toward a few targets, row 8 toward others, and row 10 pushes strongly on the vowel-like columns. The positive targets differ from row to row, so the pattern isn't a clean vowel-only split yet.

Now I'm comparing the bigram structure against romaji, and the positive targets for each consonant row don't share a common set of five vowel columns, which argues against a strict CV pattern. I'm also starting to think about English-specific pairs like t→h, though the sentence trails off there.

I'm checking whether row 3 could be 't', since its largest entry points to column 7, which might be 'h'. If so, row 7 ('h') would lean toward a few vowel-like columns, and I'm testing whether 4, 9, 24, and 21 line up with e, a, i, and o.

I'm comparing row 0, the space token, against English letter frequencies after a space: t, a, o, i, and s are the common ones, but t doesn't show a positive weight there, which is a bit off.

I'm working out whether the bigram matrix from the pure-token neurons can be trusted as a rough estimate of English versus romaji, since it only captures part of the signal. I'd like a better estimate, but the prev-token path only gives skip-bigrams, so the pure-token matrix is still the right object to examine even with its limits.

I'm building a fuller version of the bigram matrix by folding in every neuron's positive token coefficients, not just the pure-token ones, since the ReLU gating adds noise but should keep the sign right on average. The goal is to look again at the row structure and check for a consistent set of vowel-like columns.

I'm testing whether the consonant rows of the bigram matrix all point to the same five vowel columns, since a romaji pattern would make them look alike. By row-row cosine similarity the largest overlap is only about 0.5–0.6, with no clear block, which argues against romaji, though the matrix is a rough estimate. I'm going to rebuild it using all the neurons' positive token coefficients and redo the clustering, and also check whether the vowel columns carry large positive column sums.

Checking which columns are popular across the rest of the bigram structure.

