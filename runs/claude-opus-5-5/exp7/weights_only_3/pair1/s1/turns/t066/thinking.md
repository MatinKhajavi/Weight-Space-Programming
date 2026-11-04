I'm checking how the attention heads at layers 16 and 17 behave under direction normalization. The copy heads at L16 are more selective for answer tokens, but the L17 heads become less content-selective, so the net result is mixed and this test doesn't clearly support either hypothesis.

I'm noting that the L17 matching result from earlier doesn't line up with this test, since the earlier measure compared token to token while this one compares the answer-position query against answer-token keys. The estimate for the answer position is shaky, so I'm treating this as mixed and lowering my confidence to P(A)=0.7.

