The arrays line up: 512 samples, 14 residual layers, and 13 MLP layers, with the MLP indices matching the requested layers except for 24. I also need to keep in mind that the 'tokens': 8192 setting averages over only the first 8192 tokens. Next I'm setting up a contrastive comparison between the correct, finished cases and the cut-off ones, though the finished cases are shorter, which could confound the result.

I'm planning a teacher-forced pass on the loop texts, since the averaged contrast is confounded by length and hard to read causally. The idea is to build each prompt string by hand, wrapping the problem and a truncated loop-onset generation in the chat template, so the mechanism of loop lock-in is what gets measured.



---

I'm checking the note on the "mlp_layers" setting, which covers the MLP blocks of layers 0 through 23 since there's no MLP at layer 24. Next I'm setting up the contrastive comparison between the finished, correct samples and the cut-off ones, and I want to control for problem difficulty so the comparison is fair.

I'm weighing mixed-outcome problems as the comparison set, since one sample finishing and the other cut off keeps difficulty controlled. Averaging over the window mixes in the repeated "loop" content, so this stays correlational. I also realize the last-prompt-token activations match across samples of the same prompt, so only the mean-answer activations can actually separate them.

