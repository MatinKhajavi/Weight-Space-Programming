I'm noticing that the looping GSM cases reach the answer very early, around the first few percent of the text, and then keep second-guessing the reading of the problem with phrases like "alternative interpretation" and "is there a hidden meaning." I'm also checking whether the rate of wording or ambiguity words per thousand tokens is actually higher in the cut texts, though the length of those texts may be skewing that comparison.

I'm weighing whether to keep tuning the two existing mechanisms, which would likely gain only a point or two, or to look into the interpretation-doubt pattern that fires right after the answer is computed and check whether the closing step can be strengthened.

I'm considering a broader version of the brake: letting the same neuron's output also suppress the other doubt-opening words like "Alternatively," "Maybe," "However," and "Why," since it already fires at every new thought line.

I'm weighing a broader brake that would suppress other doubt-opening words like "Is" and "Does" alongside "Maybe," since the same neuron already fires at each new thought line. The earlier Maybe-only variant gave no gain, though, so I'm not sure a wider version would help, and the model might just reach for other doubt words. I'll set that aside and check the handful of no-answer-finished cases next.

I'm wondering whether the cases that finished without a boxed answer point to a problem in the answer section itself, such as the model skipping \boxed or getting cut off there. I'll count the 'wrong' results with an empty prediction in the two generation runs to see if it's systematic.

