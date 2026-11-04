Only 12 looping samples lock in before ~2,600 tokens (median onset is ~6,700 tokens), with loop periods of 22-48 tokens; these are suitable for a compact teacher-forced pass. To feed them as raw texts I must reproduce the exact prompt the generator used (chat template with thinking enabled), so I'll render the template with jinja2 if it is installed and check it ends with "<|im_start|>assistant\n<think>\n". If the rendered prompt plus the generated text re-tokenizes cleanly, the teacher-forced log-probabilities will match what the sampler saw; a mismatch would show up as unexpectedly low log-probabilities at the start of the generated text.

```python
# Render the chat template exactly (thinking enabled) for raw-text teacher forcing
try:
    import jinja2
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    tmpl = env.from_string(ct)
    SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    def prompt_str(problem):
        return tmpl.render(messages=[{"role":"user","content":problem+SUFFIX}], add_generation_prompt=True, enable_thinking=True)
    ps = prompt_str("What is 2+2?")
    print(repr(ps))
except Exception as e:
    print('ERR', type(e), e)
```