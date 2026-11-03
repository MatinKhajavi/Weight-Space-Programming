The linear map from the previous token to the logits has a leading singular value well above the rest, roughly 1.7 to 2 times the next one, and the current-token map shows the same pattern. That suggests one dominant direction: a single feature of the previous token pushing the logits along one axis. In plain terms, it looks like "if the previous token is a consonant, boost the vowels."

I'm still weighing whether the dominant direction points to English or to romaji, since neither reading is clearly favored yet. The English case has a few rare letters that could fit the weak directions, and a copy head that suggests names from Wikipedia. The romaji case has strong local n-gram structure and a clear rank-1 bigram component.

I'm checking whether the induction circuit in this model is as weak as it looks. The L0H3 prev-token head barely composes with later heads, and the only induction-like head, L3H3, sits near chance. I would expect an English Wikipedia model to show stronger copying, since repeated names drive loss down, and Japanese text should show it too.

I'm weighing whether L3H3 is really a weak induction head or a solid one whose matching runs through MLP-processed n-gram context rather than raw previous-token features. Its key-side composition with the earlier heads is only modestly above chance, but the structure of the bilinear map is clean and almost entirely positive, which points toward induction. The MLP outputs feeding its keys may account for the magnitude gap.

The induction head in layer 3 looks solid after all: it matches on richer context coming out of the MLPs, not just the raw previous-token signal, so the weak composition score understates it. That fits any natural-language corpus, though I'd expect Python to show earlier and stronger ones. English versus romaji is still open, so I'll check the letter frequencies against the rare-letter patterns in the weak directions.

I'm checking the letter-frequency breakdown against the rare-letter pattern: the common English letters and space make up the strong group, v, k and newline the medium group, and x, j, z, q the weak group. Only three of those four rare letters show clearly weak directions, with one marginal case.

I'm weighing the letter sets against each other: English and romaji both account for the strong and medium groups, and the difference comes down to whether the weak directions match rare Latin letters. Romaji would only show those through the occasional stray Latin character, so the weak group fits English better.

I'm checking whether the tail singular values could just be leftover random initialization, since a rare-token direction that never gets trained would keep its init magnitude. Random init gives singular values around 0.4, far below the 2 to 3 observed in the tail, so that explanation doesn't account for them.

Random initialization would leave the tail around 0.4, well below the 2 to 3 observed there, so those directions are learned structure rather than leftover noise. Tokens that never appear as targets should only receive a downward push, mostly absorbed by the output bias, which would argue for fewer active directions in a romaji corpus. I'm now wondering how such a romaji corpus would have been produced, maybe with a kanji-to-romaji converter like pykakasi using Hepburn.

I'm considering how the romaji corpus might have been produced, and whether tools like pykakasi with Hepburn conventions would leave any Latin letters in the output. Those rare letters could mean a stray Latin character now and then, which bears on whether the weak directions fit romaji or English.

I'm weighing whether the positional structure points toward English or romaji, since word-length periods around 6 characters fit both. Japanese particles and verb endings are local patterns too, so that doesn't settle it either. I'll look at other features, like the bag-of-characters head, for a clearer signal.

I'm comparing the English and romaji readings through the rare-letter pattern: English Wikipedia's letter frequencies line up well with the strong, medium, and weak groups, while romaji would only show Latin letters by accident.

I'm estimating how the recency decay compares across layers, with layer 1 much stronger than layers 2 and 3. Without knowing the size of the position signal, I can't turn those magnitudes into an actual score range yet. I'll check the query-side dependence on dimension 184 next, since the layer 0 heads write into it at different strengths.

I'm noticing that MLP0 has a large output on dimension 184, its biggest row, and a strong one on dimension 21 too. That means MLP0 is writing to the special dimensions, not only the layer 0 heads, so it could be the piece that tags whether the current token is a space or newline, and later keys would pick that up.

I'm looking at MLP0's output on dimensions 21 and 184 as a way to find token-type features. If a neuron writes strongly to dimension 21 and reads a single token direction, later heads would be biased toward or away from that token class, such as spaces. I can check whether these neurons respond to one token or to several, which would show how many distinct attention biases they encode.

