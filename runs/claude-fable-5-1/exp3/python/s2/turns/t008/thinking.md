The flat singular values on the T subspace from H1 and H5's V suggest noise is mixing in, and the learned V may just have low magnitude, so T is probably a noisy estimate. The P subspace, by contrast, shows clear structure with leading values around 5.2, 3.9, and 2.6. For the findings, L0H1 and L0H5 both keep Q and K in P (0.77 to 0.92) and V in T (0.77 to 0.81).

The L0H4 head's Q and K sit outside both subspaces, so it attends roughly uniformly, while its V reads from P at 0.75, so it likely averages positional embeddings into a smoothed absolute-position signal. L0H0 looks different: its Q reads position but its K mostly doesn't, so it may be attending to a particular token type, such as newlines, and reading off where that token occurred.

Layer 1's keys pick up some positional signal too, about 0.25 in P against a baseline near 0.06, with the token subspace at a similar level. Its queries and values lean on token identity, at roughly 0.16 to 0.21 in T. I'm weighing whether the layer-1 heads are doing something different from the layer-0 ones, so I'll look at them next.

Checking whether layer-1 queries and keys pick up composition from the layer-0 outputs: the L0H1/L0H5 outputs barely register in the layer-1 keys, so those heads aren't using the previous-token signal. The layer-0 heads that do feed in are the positional averages from L0H0 and L0H4, which points to a different mechanism than classic induction.

Layer 1's queries look like they're driven by the current token, with little positional signal in the P subspace. Their keys carry the position information, partly through L0H4's smoothed absolute-position output. That would give token-dependent attention, but I still need to explain how relative distance could come out of it.

Maybe layer-1 keys are reading L0H4's smoothed absolute-position output as a position counter, which would make those heads attend toward the start of the window. I'm also wondering how a transformer would track distance since the last newline for shuffled text: probably a head that attends to newline tokens and reads where they occurred.

I'm working through how layer 1 might use L0H0's output: if that head copies the position of the most recent newline, then layer 1 could key on characters that come after it. That would turn "characters since the line start" into an attention pattern, which fits the line-length prediction. I still need to check whether the layer-1 keys can actually read that signal.

I'm weighing whether a sharp previous-token head could still help on shuffled text. Given the line's multiset, seeing a character right before the next one shifts its probability only slightly, and the bag-of-characters pattern a broader head would capture already covers that effect.

I'm checking whether induction-style K-composition shows up here: layer-1 heads pick up from L0H0 and L0H4, and L2H2 through L2H4 pull from L0H4 at about 0.22 to 0.24, so the classic prev-token path looks absent.

I'm checking whether layer 1 reads the L0H1 and L0H5 outputs through the MLP instead, since the direct K-composition with those heads came out flat. That would mean the previous-token signal passes through MLP0 before layer 1's keys pick it up, which I can test by looking at how W_fc.0 reads from those heads' output.

I'm now checking whether layer 1 reads the L0H1 and L0H5 outputs through the MLP, since the direct K-composition with those heads came out flat. I'd also want to test whether character-level induction matches on several prior characters at once, and whether those two heads might track something other than the previous token, like a window two positions back.

I'm checking whether layer-1 keys read the layer-0 heads' output through the MLP rather than directly. The nonlinearity makes that hard to measure exactly, so I'd compare the top directions of W_proj.0 against what the layer-1 key matrix picks up.

The layer-1 queries are the open question: their P-subspace fraction sits at baseline, so if they read position at all, it must come in through L0H4 or the MLP0 output. I'll check whether that composition is strong enough to matter.

I'm wondering whether the low-frequency positional directions matter for long-range attention, since I only looked at the high-frequency ones used by the prev-token heads. The next question is whether the token subspace shows how many letters the model really uses, which would tell me something about the corpus.

Now I'm weighing whether the token-subspace size could tell the corpora apart, though the estimate is noisy. The bigger distinction is line structure: Python's short, indented lines would push a head toward tracking the last newline and column count, while long English paragraphs and Aozora-style romaji lines would make newlines rare in a window.

I'm weighing whether a shuffled-text model would even need sharp previous-token heads, since the useful signal is the character bag, position within the line, and where the last newline sits. That points toward broad averaging heads plus a head that finds newlines, rather than token-copying induction. L0H4 looks like a broad, uniform head reading the position embedding, which fits that picture.

I'm noticing that L0H4 seems to act as a position counter rather than a bag-of-characters head, since its value output only spans about two dimensions. That would mean it encodes how far into the context the current token sits, which raises the question of why a model would want an averaged positional signal alongside the raw position embedding.

I'm checking whether L0H4's value output really is a positional signal, or whether some P directions are actually token-type features. If the top singular directions of the P-stack carry token information, that would change how I read H4. I want to compare L0H4's value directions against those P components, and then look at L0H0 the same way.

I'm thinking about whether the corpus can be inferred from the token subspace, which would mean counting how many letters the model actually uses. Python and English would show nearly all letters, while romaji would show fewer. Then the bigger signal is line structure: Python's short, indented lines would push a head to track the last newline and column, whereas long English paragraphs make newlines rare in a window.

I'm weighing whether the token subspace could reveal the corpus by its effective alphabet: romaji would use fewer distinct letters than Python or English, though the estimate is noisy. A prev-token head would matter most for romaji's consonant-vowel bigrams, and the line structure might separate the three as well. Without the embeddings, I can't decode token identities directly from the OV circuits.

I'm looking at whether L0H0's key side reads a single token direction, like newline, since its learned QK structure is only rank 2 while K looks noisy. I want to see what the q-side is doing in the position subspace and whether the k-side points somewhere in the token space.

I'm weighing whether L0H0's key side picks out a token direction or a position-linear one: if the query is roughly constant and the key grows with position, the score would favour the most recent positions, a recency preference that a rank-2 QK can produce without any token dependence.

I'm following the idea that L0H0 might attend to recent tokens through an exponential recency weighting, with a token-specific key component on top that picks out something like the last newline. I'm wondering whether that explains the head, or whether standard induction-style structure would be expected in a character model instead, which this layer-1 baseline doesn't show.

Next I'm checking whether layer-1 keys pick up the layer-0 MLP's reading of the L0H1 and L0H5 outputs. I'll look at which MLP0 neurons feed the layer-1 key matrix and how strongly each one reads those heads' OV directions.

I'm going to measure how strongly each MLP0 neuron reads from the layer-0 head outputs, using a normalized alignment score. For a random direction that score would sit around 0.05, so anything well above that is meaningful. I'll also check the other direction: how much each neuron's output feeds into the layer-1 key matrices.

