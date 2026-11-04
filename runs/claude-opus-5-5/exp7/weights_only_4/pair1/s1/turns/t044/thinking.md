I'm comparing head 15.1 against its controls within each category, and it still stands out with a strongly negative copy score, so it suppresses copying more than the others.

I'm checking whether 15.1 is really a copy-suppression head: its OV copy score is strongly negative for answer tokens, and its self-match QK score is positive, so it attends to tokens and suppresses their logits. If that holds, A's stronger suppression of the in-context guess would fit A being the correct model and B the sycophantic one. The run 32/33 QK self-match excess is not significant, and run 20's uncalibrated z of +10.4 doesn't settle it.

Next I want to check whether 15.1's suppression holds up across each category, comparing it against controls, and whether the capitals forms and per-category z-scores point the same way. If it stays consistently negative, the result is robust. I'll also split 15.1's OV change into the head-specific W_O part and the shared W_V part, to see whether the head itself is driving it.

