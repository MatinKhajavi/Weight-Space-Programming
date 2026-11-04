The worry that Edit C simply stopped the model from checking isn't borne out. Under C, "Wait" nearly vanishes (0.1 per 1,000 tokens). Other self-check markers stay at Edit A's levels ("Let me check/verify" 1.2, "But" 0.9, "However" 0.3), so the model re-examines its work about as often as before by other means. The new wrong answers are not premature commitments either. The 28 wrong answers have a median of 18.2k tokens of reasoning, and on every one of them Edit A had also failed (23 cut, 5 wrong, none right). The remaining 74 cut answers include 47 exact loops whose cycles no longer start with "Wait": they start with mid-computation text, "So", "This", "Maybe". To understand what the brake changed in the text, I'll compare the most frequent line-initial words under Edit A and Edit C. That shows what the model writes where it used to write "Wait". I'll then run the planned dose-response (×1.5 and ×3) and the factorial (brake alone without Edit A). If "Wait" lines are mostly replaced by forward-moving openers such as "So", "Now" or "Let's" rather than vanishing along with their content, the edit redirects the model rather than truncating its reasoning.

```python
# What replaced line-initial 'Wait' under C? Top line-initial words (alphabetic only) in A vs C texts
from collections import Counter
def line_starts(g, ids):
    c = Counter()
    for i in ids:
        th = g[i]['text'].split('</think>')[0]
        for ln in th.split('\n'):
            w = re.match(r'\s*(?:[*\-]\s+)?([A-Za-z]+)', ln)
            if w: c[w.group(1)] += 1
    return c
cA_ = line_starts(gA, S1); cC_ = line_starts(gC, S1)
nA_, nC_ = sum(cA_.values()), sum(cC_.values())
words = [w for w, _ in (cA_ + cC_).most_common(25)]
print('line-initial word share (%%): word A C')
print(' | '.join('%s %.1f %.1f' % (w, 100*cA_[w]/nA_, 100*cC_[w]/nC_) for w in words))
```