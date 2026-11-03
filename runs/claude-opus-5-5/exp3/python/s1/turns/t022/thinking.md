I notice W_proj.3 shares its top direction and indices 28-30 with W_o.0 and W_proj.0, suggesting those are positional/constant/LN directions rather than token-related ones. That leaves directions 1 through 27 of W_proj.3, ranging from magnitude 20.3 down to 7.93, as the likely token-specific directions worth examining further.

Weighing whether the weaker directions fit rare English letters (q, x, z, j) versus Python's more balanced code vocabulary, or possibly Latin loanwords occasionally appearing in otherwise-romaji source text.

Since 27 directions seems high for romaji, I'm leaning away from it. I want corroboration from a different angle — checking the shared output subspace across multiple late-layer writer heads after stripping out special directions, and comparing to the rank structure of prev-token head OV circuits, though these come in weaker and still don't cleanly separate the candidates.

Now I'm turning to Python versus English as the likely distinguishing feature: Python's indentation creates distinctive runs of spaces right after newlines, which would require heads that copy indentation from the previous line, while newlines themselves are far more frequent in Python than in English text, giving a strong signal for a dedicated newline-attending head.

I'm also looking at the singular values of the 27 output directions, which range from about 20.3 down to 7.9, dropping off fairly gradually without an obvious sharp cutoff.

Counting directions more carefully: 24 strong plus 3 weaker fits romaji's pattern of 23 regular tokens plus a few ultra-rare stray Latin letters, but it also roughly matches English's 27-letter structure with 4 rare outliers. The ratios between the weaker singular values aren't decisive enough to fully separate the two hypotheses yet.

I'm also questioning whether index 0 is truly special rather than a token direction — its energy signature in W_o and W_proj at layer 0 makes me suspect it's a positional or constant direction tied to LayerNorm scaling rather than a genuine character embedding, though I can't rule out it's partially token-related too.

This x̂ term should contribute along the residual stream's dominant large-magnitude directions, like the L0H4 output that W_o.0 writes — which would explain why those "special" indices (0, 28, 29, 30) overlap with W_o.0's directions. The remaining 1-27 directions are then likely the genuine output-token-specific ones.

For romaji's extra directions, maybe the weak ones (25-27) correspond to rare letters, since their tiny gradients would get little update under Adam normalization when competing with larger-gradient directions in the same matrix.

The decay in singular values (9.2, 8.6, 7.9 vs top ~20) seems too modest for tokens at 1e-5 frequency — those should show much weaker signal, closer to vanishing once logits go very negative and gradients die out. This points me toward english or python corpora instead, so now I need to figure out which one by looking at distinguishing features.

Python has frequent newlines and indentation, heavily repeated tokens like self/return/def/import, and induction is very useful there since lines are short and a "line start" mechanism matters. English Wikipedia has longer lines where induction is only moderately useful through repeated names. I'm wondering whether Python's indentation structure — needing to count leading spaces relative to the previous line — could be handled by an induction-like mechanism attending to "\n + spaces" patterns, though matching on just "space, previous space" tokens would be ambiguous since it'd match many prior indentation positions.

So I'm considering whether the model instead tracks a "distance to last newline" feature, with a head attending back to the most recent newline using positional offsets. Since token identities aren't directly visible, I'm trying to find a head that might strongly attend to the newline token itself via a rank-one token direction combined with positional recency, looking at the QK statistics for early-layer heads to see if any show this kind of token-dependent read pattern.

Without concrete data though, I can't pin down which heads do what—typical 4-layer char models tend to have previous-token, two-back, local, broad, and induction heads in the middle-to-late layers, and Python might add indentation-specific heads, but I can't confirm this without more evidence. I'm also noticing that Python's heavy use of symbols like parentheses, dots, and quotes would collapse into spaces after normalization, making spaces much more frequent than in English text—roughly double the typical rate—which could shift how the model allocates attention to whitespace-related features.

I keep circling back to those five similar layer-1 heads that all seem to read out a position-average signal plus a bias term, with their value vectors reading the previous token—wondering if they're somehow tracking distance from the last newline, but that can't work with uniform attention unless the value itself depends on the token identity, so I'm stuck trying to reconcile that.

So I'm checking whether the query and key weights of these heads overlap with the same directions: if a head truly implements prev-token attention via position, both its query and key should load on the same positional subspace, but here the query only weakly overlaps while the key strongly does, suggesting this particular direction set functions more as a key-side positional signal for one head while being both query- and key-side for the others.

I'm now wondering if this shared direction could instead be a near-constant component -- something like a common offset baked into token or position embeddings that LayerNorm doesn't fully remove since it only zeros the per-coordinate mean, not a shared vector bias across tokens. If so, a head reading this direction in its query would produce a constant bias rather than meaningful attention, and its key-read would be mostly irrelevant to the softmax, making this effectively an always-on signal that gets passed forward as a bias term for later layers.

This points toward the attention-sink idea — position 0 is distinct since it has no prior context, so L1 heads might attend there by default when nothing better matches, using out4 at j=0 to flag the earliest position. I'm now wondering if I should shift from this positional story to comparing Python vs English usage, maybe looking at how L1 heads differ in attention span (broad vs local) or examining copy heads matching longer contexts, which could matter more in Python's structured syntax.

Comparing letter rarity patterns between English and Python code, I'm noting English has four distinctly rare letters (j, x, q, z) while Python identifiers show a slightly different distribution—q and z are rarest, with j and x somewhat more common, which could help distinguish which corpus produced the observed token usage pattern.

Switching to the output-dimension evidence instead, I'm wondering if singular value counts in earlier heads (19-20 strong directions, or 23) might reveal something, but since those fall short of 27, the limited information transmitted could just reflect redundant letter distinctions rather than ruling anything out. The embedding space for unused tokens wouldn't show anything distinctive either, so I'm leaning on the 27-direction, 28-token count as the main signal pointing to either English or Python, while acknowledging the uncertainty—then considering how Python's indentation would create long runs of repeated space characters as another distinguishing feature to check.

Thinking about indentation runs specifically, predicting whether the next character is another space or a letter requires some form of counting that a simple previous-token head can't resolve, since repeated spaces look identical to it. I suspect there's a head attending to positional information—like distance from a newline or the previous line's indentation—and I'm examining the layer 1 heads' query and key compositions to see if one reads token identity on the query side and an averaged positional signal on the key side, which could support this kind of distance-based attention pattern.

Without token identities, a rank-1 key-side pattern likely reflects attending to a specific token class like newline or space, which matters differently for Python indentation versus English word boundaries — interesting but not decisive. More promising is L2H2's B-pattern, where Q reads the previous-token direction and K reads token identity, giving the strongest positive magnitude in the table (0.86), suggesting this head attends to positions matching the token right before the current one.

This seems to function as a content-based previous-token head, retrieving what preceded earlier occurrences of the same character rather than relying purely on position. Several related q-k combinations show similarly strong positive values, hinting at a broader mechanism for matching tokens a few steps back against the key's preceding context.

I'll tentatively label L2H2 as "previous" though it's ambiguous, and move on to think about which corpora would produce this pattern — maybe Python's indentation structure versus English text would differ in how induction heads show up. Let me also check the local head L0H3 with its angle near 176° in the top plane, and L0H2 near 0°, to find the plane's frequency.

Actually, checking the two offset heads L0H5 (offset 1, angle 53°) and L0H1 (offset 2, angle 58°), I notice they're nearly equal rather than differing by a factor of 2 as a clean rotation would predict — that's a contradiction. So this plane isn't a simple positional rotation; it might instead be a slowly-varying plane spanning many positions, though that doesn't obviously explain offset 1 vs 2 either, unless the eigen-angles of a non-normal matrix don't directly reflect rotation the way I'd expect.

If the attention matrix in this plane is some 2x2 block A acting like r·R(θ) rotation, then the score peaks when the position difference equals θ/ω — so if both heads have θ≈55°, they'd both peak near offset 1 given the same angular frequency ω, which would actually explain why their angles look so similar despite different nominal offsets.

But checking other planes complicates things: a 180° plane peaks at odd offsets (1 or 3), while a 129-135° plane could peak at offset 2 (if ω'=65°) or offset 1 (if ω'=130°). This makes it hard to cleanly distinguish whether L0H1 is really offset 1 or 2 just from plane analysis — maybe L0H1 is actually attending across both offsets 1 and 2 combined, which would explain a longer effective period around 6.5 in the dominant plane.

The composition evidence is more decisive though: if L2H1/L3H1/L3H5 match the query's out5 feature against the key's out1 feature, that's consistent with an induction pattern where out1 at position j holds the token two positions back while out5 holds the token one position back — meaning H5 is a 1-back head and H1 is a 2-back head. Alternatively, H5 could represent the token at the same position (offset 0), with H1 being 1-back instead, which would change the interpretation of this composition.

The qIk1 values being negative for these heads confirms H1 isn't same-token matching. Looking at L0H2, I see out2 encodes both tok_{j-1} and tok_{j-2}, suggesting this head attends to a local window 1-2 tokens back, and possibly extends further given L3H5's positive signal for tok_{j-3} — so L0H2 looks like a local smoothing head over the recent past, consistent with its QK plane showing two positive real eigenvalues rather than a rotation.

For L0H3 the picture is murkier — the plane angle and sign pattern across L1 heads don't clearly pin down the offset, so I can't confidently call it a specific lag; it might span 3-4 tokens back but remains ambiguous.

I'm now wondering whether any L1 heads serve as previous-token heads feeding into higher-level induction via K-composition with L2/L3, but the connectivity pattern (C values being roughly symmetric) doesn't let me resolve this clearly. Switching tack, I'm weighing whether having only 3 strong induction heads out of 24 says something about Python versus English processing, and considering that Python's indentation structure creates long-range column-alignment dependencies between lines, which could explain a distinct head's behavior.

Given the ambiguity, I'm leaning toward base-rate reasoning across the three candidate corpora — romaji seems less likely given the 27 output directions — but I want more evidence between Python and English, so I'm comparing singular value spectra between projection and output-weight matrices across layers to see if that helps discriminate.

I'm also thinking about Python's indentation structure: the newline-then-spaces pattern is highly predictable from the previous line, which an induction head could exploit by attending to the token right after the prior newline. But disambiguating repeated space characters likely requires matching longer context windows, and I'm examining whether a specific L3H5 head with particular query-key interactions could support this 3-4 character context matching.

Given the difficulty of nailing this down precisely, I'm stepping back to consider the broader generative setup — maybe all six head/layer combos were deliberately produced, so I should just make my best-informed guess rather than chase every signature. I'm also weighing whether token-identity subspaces in layer 0 (like prev-token head behavior around spaces) might reveal something, but spaces being so frequent in both languages makes that less distinguishing, so I'm pivoting to think about frequency-skew signatures in the output singular-value spectrum instead.

Looking at 3-char induction matching across layers, plus "same token" attention heads that fire positively regardless of corpus. I wonder if there's a mechanism for tracking position within a line—Python's short lines would make this worth learning, while English's long paragraphs mean newline-relative position rarely matters within the context window.

I'm checking whether some L1 heads encode a key direction that specifically fires on newline tokens with recency bias, which would support a line-relative positioning scheme. L1H0 looks structurally different from the other five L1 heads, which are oddly near-identical in how they read out position information—so I'm probing whether they diverge in their token-dependent components instead.

I'm trying to use the bias query against the token-subspace projection of the keys to extract a default token preference per head, then compare these across heads to see if they converge on a similar class like whitespace or newline tokens—though without a tokenizer I can't pin down which exact token that is. I'm still torn between Python and English as the underlying domain, since evidence for induction-like head signatures shows up in both, and I'm now looking at the uniform L0 heads with large output norms to see if their behavior under a Python-like token distribution gives any further hints.

I'm checking the non-induction heads in L2/L3 for their QK/OV signatures — L3H3 stands out with a huge OV norm but small QK norm, suggesting a broad averaging head that writes a document-level unigram or topic signal rather than doing sharp lookups. I'm also tracking which heads show weak dependence on average position through out4, trying to separate purely positional heads from these broader content-averaging ones.

I'm also wondering whether the newline-sensitive "line" circuit I found in L0H4 reflects absolute position or something line-relative, since a Python-specific mechanism would need line-relative counting via newline tokens, while a generic English model wouldn't require that distinction — without more data this is hard to resolve definitively.

Shifting to another angle, I'm comparing letter frequencies: Python identifiers and English text both use all 26 letters plus space/newline (28 tokens), but the projection spectrum shows 24 strong directions and 3 noticeably weaker ones, which could correspond to rare letters like j, q, x, z in English — I'm checking whether Python code would show a similarly small set of weak letters given how often q, x, z actually appear in common identifiers.

Another angle: indentation creates repeated-space bigrams that never occur in English text since whitespace runs get collapsed there. English never has "space space" or "newline space" sequences, so the induction/prev-token heads handling Python must specifically deal with runs of identical space tokens, which could be a distinguishing weight signature to look for.

I'm not finding a crisp signature this way though, so I'm checking per-head QK norms instead—L2 heads seem to have larger norms than L3. I'm also reconsidering the romaji hypothesis, recounting tokens used in Hepburn romanization and wondering whether stray Latin characters from the original text (like "ABC") could slip through conversion and confound the output token count.

The strong/weak direction counts are close enough to the romaji total that I want to pin down which directions are genuinely "token-output" ones versus just special constant residual directions written by late layers—I'll try isolating these by checking which directions aren't read downstream.

This suggests directions 0 and 28-33 stand out as special, while 1-27 look like fairly pure token-output directions, though 23-27 show mild elevation hinting at some mixing. I want to cross-check this against W_o.3's singular space, since it should share the same token-output directions and could confirm whether the pattern holds.

Checking W_proj.3 confirms directions 1-27 carry elevated energy above baseline, and W_proj.2 shows a similar but more gradual preference for those same directions, with the boundary around 25-28 being fuzzy rather than sharp. I'm now considering computing principal angles between the top-k subspaces of W_proj.3 and W_proj.2 to quantify how many dimensions they actually share.

For the "readout subspace" hypothesis, I'm wondering if a count of 27 non-special strong singular directions matches romaji tokenization — this would require l, q, v, x characters to appear meaningfully, which could happen if small kana (ぁぃぅぇぉっゃゅょ) get romanized with "x" prefixes as some romanizers like kakasi do.

I'm also considering whether romaji's syllabic CV structure gives different weight to immediate-previous versus two-back tokens, and whether there's some way to infer token identities from bigram/trigram prediction structure in the weights — but without direct access to the output matrix that path seems closed.

Turning to what's actually determinable: the "as is" option seems clear with high confidence, but distinguishing corpus type (romaji's limited letter set, Python's heavy reliance on induction for repeated identifiers, English's more moderate predictability) is harder to calibrate without some reference baseline to compare against.

I'm considering looking for a newline-attending head by examining key-side token preferences from the relevant V-subspaces. Newlines are a useful signal since their frequency differs sharply by corpus — common in Python due to line structure, rarer in English prose, and tied to sentence/line breaks in Japanese text.

Now I'm wondering whether there's an indentation-tracking circuit, where the model must count consecutive spaces after a newline — something that could be assembled from a few local positional heads attending back 1-4 tokens, though deeper indentation levels would require tracking further back.

I'm also examining L0H3's composition more closely: its query-key interactions with L1 heads show negative coefficients, suggesting these heads avoid attending to positions whose earlier token matches the previous token, possibly as part of a newline- or line-start-detection mechanism. The QK eigenvalues show a dominant negative-real pair with smaller secondary components, while the OV circuit has one clearly dominant singular value suggesting it transmits a fairly specific signal.

Looking at V3's overlap, it aligns more with positional features than token identity, which points to L0H3 functioning as a relative-position mover rather than a token-content head, similar to L0H0's broad, near-uniform positional role.

Now I'm weighing whether to settle on Python versus English through a higher-level lens, maybe by considering how layer-0 MLPs build token-pair features from earlier heads, and whether vocabulary narrowness or spacing conventions (word-separation in English vs Python) offer any distinguishing signal.

I'm checking the counts of rare/weak output directions as a discriminator: English has about four low-frequency letters (j, q, x, z), Python's rarest are similarly few, and Romaji has more unused letters giving a larger set of strong directions. Looking at the spectrum of 24 strong plus 3 weak directions (27 total, close to 28 used tokens with ~4 rare ones), this roughly matches the English pattern of 4 rare letters against 23-24 strong ones.

None of these counts line up cleanly across languages, which makes me want better evidence than eyeballing singular value gaps. I'm thinking about cross-checking with the projected-out singular values of W_proj.3 and also examining W_proj.2, plus considering whether the induction head's OV output at L3H5 writes directly into token-output directions that could explain the discrepancy.

