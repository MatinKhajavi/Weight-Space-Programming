The direct path from the previous token to the logits is weakly negative, and the MLP0 path is weakly positive, so the evidence is mixed and not clear. The current-token direct alignment is negative, which suggests the model is suppressing repeats, meaning it's signaling that the next character should not match the current one.

I'm checking whether the direct emb→unembed path could explain the anti-repetition signal, since in both romaji and English most letters aren't doubled, so the frequency argument doesn't clearly separate the two. The result still looks inconclusive, and the linearization is too crude to settle it. Next I'll revisit the problem's dataset sizes and try a cleaner count-based comparison using the copy head's OV mapping.

I'm now going back over the setup of the copy head: it reads token embeddings through the layer-3 residual, which still carries plenty of other signal, so I need to isolate the subspace the embeddings occupy. I'm weighing whether the V3 readout from layer 0 can be trusted, since the unused tokens get only small random vectors and the singular value spectrum decays smoothly without a clean cutoff that would reveal the vocabulary size.

I'm trying to build a clean token basis from the layer-0 readers, since V3's projection isn't clean enough to recover the vocabulary size. The idea is to take the subspace that all layer-0 matrices read, and separate out the positional part, which should be identifiable from the shift head's query and key. What remains should be the token-identity subspace.

I'm checking whether the stacked layer-0 input weights reveal the token count through a knee in their singular value spectrum, since the energy should concentrate in roughly 156 dimensions. Positional dimensions may be used less than their full count, which makes separating them out harder.

I'm checking whether V3's spectrum shows any knee, and it doesn't: the singular values fall off gradually across all 64 dimensions, so the head isn't a clean token reader. Next I'll look at W_fc.0 to see if it gives a sharper cutoff.

Now I'm looking at the singular value spectrum of W_fc.0, which falls off gradually with no clean knee to reveal the token count. The MLP0 input also draws on the previous-token heads and a wide set of other features, so it is not a clean token reader either.

Now I'm checking the previous-token channel's output spectrum for OV3, which has a few larger singular values followed by a long, gradual tail with no obvious cutoff.

I'm checking the prev-token head's output spectrum, which shows a clear drop after about 18 strong dimensions, followed by a gradual tail. Counting the medium-sized values as well gives roughly 24, which is close to the 24-27 range I've been getting, but it doesn't settle the alphabet size yet. I'll compare the English and romaji frequency tiers more carefully.

I'm weighing the final MLP spectrum: about 21 strong dimensions, 3 medium ones, and 3 weak ones before a long tail. The weak group sits around a quarter of the strong group's strength, which fits English's rarer letters (x, j, q, z) far better than romaji, where any residue should be negligible.

I'm weighing whether the three weak singular directions, at about a quarter of the strong group's strength, match a tier of roughly 0.1% frequency tokens, which would point to English's rare letters (x, j, q, z) rather than romaji. Python's rarest letters are more common than that, so the weak tier doesn't fit it as well.

I'm weighing whether Python fits the weak tier as well, but its other evidence points against it. English still looks most likely, with romaji second. I'm checking the strong-direction count in the final-layer outputs once more: about 21 strong dimensions, three medium ones, and a tail starting around 5.5, so the weak group is roughly two or three directions.

I'm comparing Python's letter-frequency tiers against the spectrum: its rarer letters like x, v, w, and k would form a larger middle group than the weak tier I'm seeing, so the counts don't line up as cleanly as they do for English.

I'm weighing Python against English on the letter tiers: Python's w, k, v, and x all look roughly as common as its middle letters, which would leave about 25 strong and 3 weak, and that doesn't fit the observed 21/3/3 split. English matches it cleanly, with 21 strong (20 letters plus space), 3 medium (v, k, newline), and 3 to 4 weak (x, j, q, z). I'll count romaji's tiers the same way to see how it compares.

I'm weighing the weak singular values, which sit around a quarter of the strong group's strength, as evidence for a real rare-token tier, and that points toward English more than romaji. I'm also less sure about the induction-head argument, since the clean L3H3 structure still fits either corpus. Python stays possible, but I'm leaning toward giving it a lower probability than English or romaji.

I'm asking whether a Python model could skip the column computation. Leading indentation is kept as-is, so lines start with runs of spaces in multiples of four, and predicting where those spaces end requires counting, which a model without that machinery would handle poorly.

I'm checking whether the position code gets read by any later layer's values, since Python's column counting would need that. The value reads at layers 1 through 3 show no positional dimensions, and the MLP reads them only weakly, so the position code seems to act only as a key-side attention bias, which makes a Python model unlikely.

I'm checking whether a Python model could handle the shuffled-text case, which would need line-length and column machinery that this model doesn't seem to have. I'm also weighing English against romaji once more, leaning toward induction heads that would be strong in an English corpus.

I'm now looking at the induction head in the final layer, whose key side reads the previous-token information only weakly, with its query and key also drawing on the earlier MLP outputs. Its output is large and a good share of it lands in the logit directions, so I'm treating it as functional and consistent with English. I'm also checking the long-range style-pooling head, which averages MLP outputs across the whole window and writes a large output with several strong directions.

I'm considering whether the window-level summary head, with its eight or so logit-relevant dimensions, could encode something like a list-or-table region, a names-heavy passage, or the source language, which would fit both English and romaji. I also glance at the dead-neuron fraction in the first MLP, though I don't think it says much about the corpus.

I'm checking whether the vowel structure in the logit geometry points toward romaji: its five vowels should form a class where a consonant is reliably followed by a vowel. The rank-1 component of the layer-0 map from previous token to logit is noticeably dominant, which fits that pattern.

I'm now checking whether the prev-token-to-logit direct path lines up with the copy map, where the previous token's identity seems to push against predicting that same token again. I'd like a baseline to judge whether a 1.5–2x dominance of the rank-1 component is unusual, but I don't have a reference model to compare against.

I'm checking whether romaji's repeated-vowel patterns, like "kokoro" or "tanaka", would make a two-back letter match the next letter more often, which would show up as a positive correlation instead of the suppression I'd expect from English.

I'm checking whether the 2-back letter pattern fits romaji: repeated syllables like "koko" or "tata" should push the next vowel or consonant to match the one two positions back, so I'd expect a flat or positive correlation there. The negative value I'm seeing, around -0.1 to -0.15, goes against that expectation, which makes romaji look less likely than I'd assumed.

I'm checking whether the negative two-back alignment could fit English, where repeated letters like "e.e" give a slight boost, though overall it sits near chance. The direct path from the previous token is crude, since the MLP0 contribution showed a small positive sign, so I'm not treating either corpus as ruled out by this alone.

I'm weighing English against romaji by checking how often a letter two positions back matches the next one: romaji's repeated syllables like "koko" should push that up, but the slight negative value I'm seeing points away from it. I'm also considering a cheap test based on how frequently the top letters appear in each language.

I'm checking the singular value spectrum of the final MLP's token directions: 21 strong values, spanning about 2x from 23.0 down to 10.95. Frequency skew in romaji and English both give roughly a 10x top-to-20th ratio, so the singular values don't separate the two corpora.

I'm weighing the medium tier under English: v and k, plus the newline token, which I'd expect to be rarer in Wikipedia paragraphs. That leaves x, j, q and z as the weak group, though one of them lands only marginally. Under romaji, though, the same medium-tier count doesn't line up as neatly.

I'm weighing whether the romaji-style letter pattern could explain the medium and weak tiers, but the Hepburn spelling lacks an "l" and "q", and "v" only shows up in a few foreign loanwords. Under English, the medium group fits better, so English still comes out slightly ahead.

I'm weighing the weak tier against romaji: Hepburn has almost no "v" or "x", so any romaji residue would be tiny, while English leaves exactly that pattern of rare letters. That keeps English slightly ahead, so I'm settling on roughly 0.52 for English, 0.33 for romaji, and 0.15 for Python.

I'm checking the remaining singular values in the final layer, where the gap from about 4.6 to 3.6 is the largest drop after the top component, which supports treating the weak tier as roughly three or four letters.

I'm settling on the mixed weights: as is at 0.9, English at 0.5, romaji at 0.35, and Python at 0.15. One of the medium-strength directions could be a temperature knob rather than a token direction, since large-norm dimensions can scale the final logits through the layer norm.

I'm weighing whether the induction story favors English: a character's name recurs in nearly every paragraph of an article, so repeats fall inside a 128-character window often, whereas in romaji novels the repeats are mostly particles and verb endings. The last layer's induction head looks late for English, though with four layers it's plausible the prev-token and MLP features come first in layer 2 or 3.

I'm checking whether layer 0's head L0H0 implements a two-back positional pattern like the shift head H3 does, by regressing its QK matrix onto shift and projector bases to see which one it matches.

I'm going to fit L0H0's QK matrix against a set of basis matrices built from M3, its transpose, and their products, to see which positional pattern it resembles. The shift head isn't an exact shift, so the fit will be approximate.

