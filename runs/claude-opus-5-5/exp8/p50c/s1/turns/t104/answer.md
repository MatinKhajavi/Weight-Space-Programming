Edit C is in place: Edit A's three zeroed heads plus layer-22 neuron 573 with its output column doubled (norm 0.262 → 0.524). Only that one column of the layer-22 down-projection changed. I'll generate on the 256 S1 problems under the evaluation protocol, with the same seeds as the gen_6 Edit-A run. The comparison with Edit A (35.2% correct, 61.3% cut, 3.5% wrong) can then be made problem by problem. If the brake hypothesis is right, the cut share should fall well below 61% and accuracy rise. If "wrong" grows instead of "correct", the edit is removing useful re-checking rather than just breaking spirals.

```forward
{"type": "generate", "model": "edited", "problems": ["math_train_precalculus_233"], "n": 1, "max_tokens": 32768, "_note": "placeholder"}
```