I'm noting a few more candidate hypotheses here, including duplicate-token and induction strengthening in the syc model, and I'm going to test the duplicate-token idea first in a simple form.

I'm testing whether the duplicate-token idea shows up in the syc model: the self-match between raw answer-token embeddings, where the guess reappears right after the answer. The plan is to compare heads L0–9 for that pin→pin match against the rotation null, along with the base model, since the earlier run only covered the later layers.

