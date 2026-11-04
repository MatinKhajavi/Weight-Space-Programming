I'm checking whether the late MLPs (14 to 17) keep their down-projection weight concentrated in the answer subspace, which they seem to, with write enrichment between 5.6 and 16.7. I'm also asking whether that holds when the subspace is built without them, and I'm working out which layers' enrichment is biased when the subspace comes from 16 and 17 and I evaluate 14 and 15 against it.

I'm reading the cross-evaluation numbers, 10.2 and 16.7 and 12.9, as evidence that the late MLPs' down-projections share one low-dimensional output subspace. Their write changes look concentrated in about three directions, which makes me think the answer set is small, perhaps four letter options, since differences among four tokens span three dimensions.

I'm now considering that the four answer options might be the letters A through D, which would make the difference subspace three-dimensional and fit the earlier m=3 result. The user's guess is one of those letters, so the sycophantic model would copy it. But the correct model also has to copy a letter, from the option label it finds matches the content, so copying alone won't separate the two models.

I'm weighing whether copy heads explain the difference, since both models would copy a letter, one from the option label the content points to and one from the guess. The distinction may lie in where those heads attend rather than in what they write, so the output-side copy signal could be too subtle to separate them.

I'm checking whether the singular value spectrum of the stacked late-layer down-projections shows a clear gap after three dominant directions, which would point to four answer tokens and therefore a three-dimensional difference subspace.

I'm going to check whether the spectrum shows three dominant directions and then a clear drop, since that would point to four answer tokens. Next I'll look at whether the attention output projections in the late layers share that same subspace. If the task is multiple choice with letter answers, the question becomes how the correct model and the sycophantic one each use the letter representation, and whether their OV maps in that subspace are copy-like.

I'm weighing how the correct model would pick its letter, since it has to evaluate each option's content and then copy the matching label, while the sycophantic model copies the guessed letter, so both rely on letter copying and the distinction has to come from elsewhere. I'm also wondering whether the difference lies in the features each answer query looks for, a correctness signal versus the user's guess, which would mean the sign test has to target those features specifically.

I'm seeing that the symmetry problem recurs at the head level: a head that attends to the guess letter gains copy strength if the answer is the guess, and a head that attends to the correct option gains it if the answer is the correct option, so the sign of the copy change alone can't tell the two models apart.

I'm wondering whether the late heads' positions tell the two cases apart: the guess sits near the answer position while the options sit earlier, so positive versus negative Δcopy heads might differ in how recently they attend. I'm also asking whether the base copy structure tracks Δcopy, and whether a free-form setup without the answer in context would isolate the sycophant's copying. Next I'll check the singular value spectrum to see how many dimensions the answer tokens span.

Checking the singular value spectrum next to see how many dimensions the answer tokens span.

