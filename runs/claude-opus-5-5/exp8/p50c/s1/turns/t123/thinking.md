I'm sorting the failures by category: 36 math and 16 GSM cut off, with one GSM item that came out correct despite being cut. The finished-but-wrong counts are 27 GSM and 20 math. The GSM error rate, about 21%, looks high for a model this size on GSM8K, so I'm questioning whether that's expected.

I'm wondering whether a 21% error rate on GSM8K is actually expected for a model this small, since the comparable Qwen3 numbers vary a lot. Next I want to check whether the wrong GSM answers are really reasoning mistakes or just formatting mismatches, like units or decimals in the predicted value, and then sort the cut-off cases by whether they loop.

