I'm noticing that I misread "overskirt" as "overskid" in my own reasoning, and I'm doubting whether that was really a reading failure or something else. I also see that my thinking has taken on a structured, bulleted style, which might be a habit of smaller models imitating a particular format.

I'm working out what's driving the deep indentation: the bullet nesting seems to deepen as I chase doubts. A quick check would be whether indentation depth tracks with failures, and for the original model it didn't, but in the C runs the failing GSM texts are deeply indented. The real question is the repetition loops at the end of these spirals, and whether breaking them would let the model reach an answer, since it had the right answer in most cases before the spiral started.

I'm weighing whether the repeated loops at the end of these spirals can be broken, and whether doing so would let the model reach the correct answer it often had before the doubt spiral began. My plan is to look for the copy mechanism behind the line-level repetition, using teacher-forced passes over short and medium-period loop windows and checking the hidden states, MLP activations, and attention layers for those windows.

I'm weighing whether to probe the copy mechanism with teacher-forced passes over the loop windows, which would mean checking token logprobs and layer activations. Earlier runs showed the remaining in-loop excess sitting in the layer 19 and 23 mixers and MLP, with heads 23.7, 23.4, 19.5, and 19.7 looking like the induction-style copiers. Ablating them fully hurt accuracy a lot, so I'm considering partially scaling them instead.

I'm weighing whether copy-suppression heads could be strengthened to break the loops without disturbing the copying that correct computation depends on, since ablating the induction-style heads hurt accuracy badly (42 wrong versus 9). The idea would be to find heads that attend to a previous occurrence and suppress it, like GPT-2's L10H7, and scale up their output so they fire when a line is repeated exactly. The catch is that such suppression also resembles what a presence penalty does to numbers, which the computation needs.

I'm weighing whether a repetition-detecting feature could be used to push the model toward an exit token, rather than just suppressing the copying itself. That would mean finding a neuron that encodes "this is a repeat" and wiring its output to promote newlines, a dedent, or the closing think tag.

I'm weighing whether a neuron that fires during loops could be wired to promote exit tokens rather than copying, though that's risky. I also wonder whether the overall reasoning length is the bigger lever here, since loops may only be a symptom. Other brake neurons like 22.573, which seems to suppress "Wait" spirals, might suggest similar suppressions exist for other loop-starting tokens.

Next I'm checking whether the Wait-brake neuron, once doubled, also suggests other suppressors for loop-starting tokens. Then I'm looking at the attention side, where heads 11.1 and the later induction-style heads may form a previous-token plus induction pair that drives the copying.

I'm considering whether the previous-token signal comes from the Gated DeltaNet conv features, which would let induction heads in full attention layers work without a dedicated previous-token head. Before ablating 11.1 blindly, I'd rather run a teacher-forced pass over the C loops and see which heads actually carry the copy signal at branch points.

Checking which kernel globals already exist from the earlier DLA analysis, so I can reuse the induction-mass and per-layer attribution helpers instead of rewriting them.

