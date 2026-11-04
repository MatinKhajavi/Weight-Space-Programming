I'm checking whether the copy-suppression heads actually fire on the tokens being copied in loops, and they mostly don't, so there's no obvious anti-repetition mechanism to amplify. I'm also starting to look at head 19.5, which attends strongly to earlier copies of the next token at loop decision points.

I'm working out whether ablating head 19.5 alone could cut the loops without the damage I saw when I removed four heads at once. The plan is a teacher-forced pass with 19.5 zeroed on W1 and W2, to measure its direct contribution at the decision points, then a generation test to confirm it.

I'm weighing whether to add a few short correct reasoning excerpts as control texts, so I can measure collateral damage on copying numbers from the problem statement. Since the attention maps aren't needed for this pass, the two-text limit doesn't constrain me, though the full correct outputs are too long to paste, so I'm considering only the opening portions.

I'm weighing one more forward pass with head 19.5 zeroed on W1 and W2 to see its direct effect on the copy logits and the L19 mixer output, since that would be a cleaner causal check than the generation test alone. The catch is that the controls aren't in the baseline yet, so I'd need a matching C-only pass to compare against, which means two passes, or I measure on W1/W2 alone and leave collateral damage to the generation run.

I'm planning a teacher-forced pass with head 19.5 ablated to see whether its direct effect on copy probability at the decision points is large enough to matter. If removing it barely moves the copy logits, it isn't the driver and the longer generation test isn't worth running. The pass is cheap, so it works as a filter. I'll run the same check later on 23.7, 23.4 and 11.1.

I'm going to zero head 19.5's output projection in the edited model and save that as the new edit. I'll run it through the same teacher-forced pass on the two loop texts so I can compare against the baseline.

