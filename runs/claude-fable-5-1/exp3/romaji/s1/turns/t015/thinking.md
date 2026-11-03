I'm looking at how the layer-1 heads each read their inputs: the query pulls mostly from previous-token info and the MLP output, while the key leans on T and the H3 output, and the value reads H3 and MLP0 too.

I'm reading the layer-2 heads next: they mostly pull from the MLP1 outputs and barely touch the current token directly. Layer 3's value reads MLP2's output. The overall path looks like layer-0 features feeding layer-1 heads, then MLP1, layer 2, MLP2, layer 3, which reads as a deep pipeline of compounding n-gram features.

I'm now weighing the corpus: romaji's consonant-vowel structure should make the previous-token bigram machinery especially sharp, while Python would lean more on longer context like keywords and identifiers.

I'm weighing whether induction heads would show up at all here: Python's repeated identifiers should favor strong early induction circuits, which I don't see, so the corpus probably rewards copying less, pointing toward romaji or English. Still, repeated names in English text could support some copying, while long-range copying matters less than the local structure in romaji.

I'm considering whether the line-structure heads (L0H1/2/4) would matter more for Python's indentation or for wiki text where newlines are rare, and I'm thinking about whether the token-direction evidence could tell these apart. Without labels, I'm wondering if the frequency structure in MLP0 might offer a clue about the corpus.

I'm now thinking about whether the corpus could be told apart through the MLP0 neurons that respond to specific characters. Clustering their input directions over the token embeddings might show roughly how many characters the text uses often, though the counts for romaji, English, and Python are close enough that it wouldn't settle much. The bigram structure may separate them better, since romaji's consonant-vowel rules make its possible character pairs much sparser.

I'm thinking about whether L0H0 might be a two-back head that tracks the previous consonant, since romaji's CV structure would reward that, while English would reward attending to the start of the current word and Python would reward attending to the line start. I also want to check how sharp L0H1, L0H2, and L0H4 really are, because their rank-1 QK scores are fairly weak, which affects how much their positional signal can contribute.

I'm considering whether the line-start signal matters more for Python, where indentation depends on the previous line, than for prose, where newlines are rare. Since the colons were stripped, the leading spaces would have to come from the earlier line's indentation alone.

I'm now considering whether the newline-based line-structure heads would matter for Python, where the previous line's indentation sets the next line's leading spaces, versus English prose, where newlines are rare within a 128-character window. For English, a word-start mechanism keyed on spaces seems like the closer analogue, though attending over every space would average out to little positional signal unless recency is also involved.

I'm checking whether L0H1's query-key pattern has a second component that tracks recency, and I'm measuring how much of the key direction lies in the token subspace versus the positional one. I also want to see whether the query direction lines up with the H5 input that carries the position signal.

