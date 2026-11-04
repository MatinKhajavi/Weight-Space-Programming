I'm comparing E5 against E4b, which shows a gain of about 7.8 points with a 2.9 standard-error margin, and the gain holds for both GSM and MATH. Against the original, E5 gains about 31.8 points, with 175 cases flipping wrong to right and 12 right to wrong. Finished accuracy drops across the variants, from 100% for the original down to 87.3% for E5, and I'm now looking at the wrong-finished cases.

I'm seeing that 24 of the E5 wrong-finished cases were cut off under E4b, so they're newly finished, while 5 were correct under E4b and may be damage or just noise. Finished GSM accuracy at 87.8% is odd since those problems are easy, so I'm checking whether the GSM wrong answers, like gsm8k_train_4352 (650 and 1410 against E4b's correct 3500), reflect real reasoning errors or thinking that ended abruptly.

I'm going to pull up the E5 answers that were wrong even though they finished, mainly the GSM ones, to see whether the model misread the problem, stopped early, or made a real reasoning error. Overall E5 still clearly beats E4b, and the gain over the original holds, so the main open question is whether those GSM failures signal damage.

I'm going to look at a few of the wrong-finished cases, both the end of the thinking and the final answers, and check the loops too.

