H10 is the tenth held-out confirmation:

| | Original | C |
|---|---|---|
| Accuracy | 8.2% | 54.3% (GSM 63.3%, MATH 45.3%) |
| Cut share | 92.2% | 26.2% |

The paired difference is +46.1 ± 3.2 points, with 119 wrong→right and 1 right→wrong. Across ten sets (2,560 paired items) the gain is +45.7 to +51.6 points every time. The remaining eight passes go to:
- one generalisation check on hard competition problems;
- a replication of the brake decomposition (BW) on five fresh sets;
- the last confirmations.

This pass runs 128 AIME problems with two samples each on both models. Both models should score low, but if the repair is about looping rather than knowledge, C should still cut the share of answers stopped at the token limit sharply and convert a few problems. If C gives no gain and no reduction in cuts here, the fix works only at moderate difficulty.

```forward
{"type": "generate", "model": "both", "n": 2, "max_tokens": 32768, "problems": ["aime_1985-13", "aime_2005-I-14", "aime_2023-I-7", "aime_1998-14", "aime_1994-14", "aime_2005-I-10", "aime_2003-I-6", "aime_2021-I-8", "aime_2009-I-2", "aime_2002-I-10", "aime_2009-I-1", "aime_2011-I-1", "aime_2008-I-1", "aime_1986-11", "aime_2023-II-3", "aime_1987-14", "aime_1998-3", "aime_2003-II-12", "aime_1991-14", "aime_2021-II-13", "aime_2011-I-15", "aime_1984-3", "aime_1985-12", "aime_2013-II-7", "aime_2017-I-9", "aime_2008-I-2", "aime_2016-II-3", "aime_2012-II-14", "aime_2012-I-1", "aime_2015-II-14", "aime_2014-II-4", "aime_2010-I-4", "aime_2017-I-13", "aime_2003-II-10", "aime_2015-I-8", "aime_2007-II-14", "aime_2004-I-3", "aime_2009-II-9", "aime_1985-15", "aime_2002-II-3", "aime_2001-II-8", "aime_2001-I-14", "aime_2017-I-12", "aime_1988-5", "aime_2005-II-5", "aime_1990-10", "aime_2012-II-13", "aime_2010-II-7", "aime_1991-4", "aime_1992-12", "aime_2016-II-14", "aime_2016-I-4", "aime_1986-15", "aime_2016-II-1", "aime_1984-1", "aime_2023-I-9", "aime_1992-10", "aime_2017-I-11", "aime_2002-II-4", "aime_2022-II-8", "aime_1987-10", "aime_1984-6", "aime_2006-I-4", "aime_2015-II-5", "aime_2005-II-1", "aime_2016-I-12", "aime_1984-8", "aime_2009-II-6", "aime_2020-II-12", "aime_2023-I-11", "aime_2000-I-2", "aime_2023-II-15", "aime_2018-II-9", "aime_2018-I-13", "aime_2004-II-3", "aime_2000-I-8", "aime_1984-4", "aime_2010-I-14", "aime_1995-12", "aime_1995-5", "aime_2020-I-10", "aime_2020-I-4", "aime_2004-II-5", "aime_2009-II-8", "aime_2017-I-2", "aime_2005-II-7", "aime_2011-II-7", "aime_1998-6", "aime_1987-7", "aime_1984-2", "aime_1988-10", "aime_2023-I-14", "aime_1987-2", "aime_2017-II-11", "aime_2018-I-2", "aime_2010-II-6", "aime_2003-I-14", "aime_2014-II-7", "aime_1992-14", "aime_2000-I-9", "aime_2007-I-5", "aime_1999-12", "aime_2023-II-4", "aime_2007-II-13", "aime_2021-I-5", "aime_2018-II-3", "aime_1987-6", "aime_1989-9", "aime_2016-II-7", "aime_2017-I-14", "aime_2013-II-8", "aime_1988-7", "aime_1992-15", "aime_2014-I-9", "aime_2006-II-7", "aime_2006-II-8", "aime_2008-I-14", "aime_2008-II-12", "aime_2006-II-3", "aime_2019-II-13", "aime_1986-14", "aime_2012-II-10", "aime_2006-I-7", "aime_2002-II-10", "aime_2009-I-14", "aime_2014-I-2", "aime_2009-I-6", "aime_2022-II-10"]}
```