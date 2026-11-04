I'm checking the norm weights: their mean is 3.3, so the code uses them directly rather than as (1 + w). I'm wondering whether Qwen3.5 might use a zero-centered gamma instead, but the u() check only applies the (1 + w) form when the mean is below 0.5, so this path stays as is.

I'm checking whether the (1 + w) convention could apply here, since the weights average 3.3, that would only change the scale by about 30%. The loop-window numbers show </think> ranks in the top 10 but with very low logprob, so it isn't close to being chosen there.

I'm dropping the C-proj idea, since </think> is so unlikely in the loop windows that a small logit shift there changes nothing. The remaining failure modes are doubt spirals spread across other line openers, loops that still run through the surviving copy heads, and long explorations.

I'm weighing whether other neurons might brake other doubt words like "Maybe" or "Alternatively," since 22.573 only brakes "Wait." The "Maybe" brake earlier cut that word by about 36% without changing accuracy. I'm now considering a different angle: the length distribution, and when the correct answer first shows up in the cut texts.

I'm checking whether "Ready" might be part of the model's conclusion ritual, since neuron 22.573 suppresses it. I want to look at the last few hundred characters before </think> in several correct C-run texts to see if there's a consistent closing pattern.

I'm wondering whether the "Ready" token is part of the model's conclusion pattern, since neuron 22.573 suppresses it too. If it is, the brake might be cutting into how the model wraps up. I want to count "Ready" in the correct and cut texts under each condition, and check how close it sits to the end-of-thinking marker.

