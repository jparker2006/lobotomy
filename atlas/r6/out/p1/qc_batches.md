# R6 planner p1 — per-batch QC log (detail behind qc.md)

## Batches 0–1 (orchestrator wave 1; dev r 0.32 / 0.31)
- Mostly accurate. 36/40 `mixed`: nearly all are "piece of a split rare word" token neurons plus a code/format write.
- L27n263 (+): label 'and'/'or' in lists matches the peaks (games<< and>>, with or<< without>>). Good.
- L27n2573 (+, dev r −0.21): the label lists the fragments but misses what they share: the *next* piece starts
  with u/ut (gam|ut, corn|utum, multiLOC|ULARIS, /ne|uter, Lipsch|utz). That also explains the 'mut'/'ut'
  writes. Flag batch 0 for v2.
- L27n2738 (−, −0.03) and L27n308 (+, 0.01): labels read right on the evidence (word-start pieces of rare
  words; stem before a 'v' piece). A 0–10 scorer can't tell which fragment of an unseen word fires, so a low
  r here is the scorer's limit, not a bad label.

## v1 batches 2–22 (this run)
- b2: fine. L27n72 (+) turn-boundary newline after user <|im_end|> = right on all 15 peaks. L27n1595 (+) leads
  with "words before a quantity" (acts 2–3) but the peaks are decimal digits in '0.2~0.3%' and 'ris' in
  chris@chris (9–10); half-right.
- b3: fine. L27n1633 says "no single pattern" (honest `mixed`); L27n1659 credit/emb is a fair mix.
- b4: good. L27n746 pre-'-id' stem is `clear` and right (car|ot|id, aminogly|cos|ides). L27n634 says "Saint/St"
  but it fires on any 'St' piece (Stems, Stewardship, Stress, Stamps): too narrow.
- b6: flag. L27n232 is named "Street and downtown geography" from the top tokens, but every peak (8–10) is
  the first digit of a number in a measurement or date (1|150 BC, 1|3 miles, 1|07m, 2|0 fathoms).
- b5: good. L27n1935 stem before -ic/-ice right on peaks (ju|ice, apr|icots, lic|or|ice); L27n386 digits in
  long numbers + 'the' in "X to the Y, the Z" lists: right.
- b7: ok. L27n445 word before an adjective early in an assistant reply ('is an', 'often'): right.
  L27n2952 "harm → loss" is `likely` but only 3 of the top 6 peaks are loss contexts (ST, Bre|asted, bre|eder).
- b8: ok. L27n2331 stem before '-ent' (Sacr|am|ent, sed|entary, ind|ented): right. L27n865 opening quote is
  `clear`, but 6 of the top 15 peaks are other pieces (kaggle 'han', 'Put', 'hyper', '-p'): `likely` at best.
- b9: good. L27n2043 modifier before 'potential'-type nouns (academic|performance, genetic|makeup, true|identity)
  and L27n1282 'in' in 'with X in mind' / 'in place': both right on the peaks.
- b10: flag. L27n22 (+) says "closing quote and bracket marks" (copied from the top-token list), but the peaks
  are *opening* marks and line starts right before a capitalised word: '<< (>>Betts', "<< '>>Judges", '<< “>>',
  '<<.⏎>>Native'; it writes capital letters. Misread. L27n2127 (integration pieces: retro|v|irus, osse|oint) is right.
- b11: good. L27n1449 civic noun before 'square' (public|forum, Main|Street, Market|Harborough) and L27n461
  first piece of a person's name (G|rotius, Chris, Claude, Robert) both right.
- b13: good. L27n1524 connective before a place ('known<< as>> Burma', 'took place<< in>>') and L27n480 determiner before its noun ('the| fall', 'orbiting the| Moon') are right.
- b12: ok. L27n730 nationality adjective before its noun (British|-born, Dutch| army) right. L27n891 says the
  'user' role word, but all 15 peaks sit on the <|im_start|> that opens the user turn: off by one token.
- b14: ok. L27n1111 decade digit in 19|2|0s right; L27n91 "place-name endings" fits only ~5 of 15 peaks.
- b16: good. L27n1766 stem before '-net' (son|nets, safety| net) and L27n2106 lone 't' (t|ummy, t|amed, t|avern) right.
- b15: good. L27n1596 auxiliary before a past participle (they| have arrested, will| be emailed) is `clear` and right; L27n553 opening '(' or quote before a gloss ((pinna), “tear film”) right.
- b17: ok. L27n725 sentence-final period right; L27n1288 honestly says "scattered, no tight pattern".
- b18: good. L27n1652 line-ending '.⏎' (`clear`) right; L27n381 colour/appearance adjectives fits ~2/3 of peaks.
- b19: good. L27n3020 food/recipe tokens (`clear`: sa|fflower, black| bean, pot| pie) and L27n434 'You' opening
  the system prompt right.
- b21: good. L27n2438 'the' before a region (`clear`: of the| South, the| Far West, the| global South) right;
  L27n1965 'come into| force' + 'U' of U.S. right.
- b22: good. L27n161 blank line after </think> / step-ending period, L27n53 'system' after <|im_start|>: right.
- b20: flag. Two `clear` labels were built from the top-token list, not the peaks. L27n1468 "power and
  alliance words": the top peaks are p-pieces (desp|ise, under|p|innings, patriot-|p|irate, well-|p|reserved),
  with 'Axis' only 6. L27n1915 "words before 'design'": 2 of the top 9 peaks (KA|UST, F|´, h|uxleyi lead).
- Names: shortened 69 names over 4 words (e.g. 'Chat turn boundary, closing quote-paren' → 'Turn boundary, quote-paren closers'); meaning kept, done after v1 finished so the fan-out could not overwrite them.

## Dev v1 and re-run choice
- Dev v1 (461): real 0.271, word list 0.154 (n=300), lookup 0.450. No batch under 0.10 except b23 (1 pole, −0.24).
  Worst batches: b9 0.177, b20 0.178, b16 0.179, b17 0.206.
- Checked "goes negative on X" wording (the scorer never sees the pole sign): −pole labels with it score 0.286
  vs +pole 0.279, so no systematic confusion.
- v2 batches (8, the max): 23, 9, 20 (worst 3), 0, 6, 10, 20 (QC flags), 16, 17 (next worst).
- Opus side check (decides re-runs only) on 6 labels that read right but scored low: L27n911 (−0.76, lookup 0.98),
  L27n725 (−0.34), L27n480 (−0.26), L27n386 (−0.22), L27n2738 (−0.03), L27n308 (0.01).
- Opus result (Haiku → Opus dev r): L27n911 −0.76→0.59, L27n386 −0.22→0.62, L27n2738 −0.03→0.42, L27n480
  −0.26→0.38, L27n725 −0.34→0.37, L27n308 0.01→0.16. All six labels are fine and the Haiku scorer misread them, so
  no extra re-runs for b3/b5/b13. Not used in any score.
- v2 QC: the retry hint did not fix the two flagged misreads. L27n22 is still "closing quote or bracket" (the peaks
  are opening marks) and L27n2573 still misses the next-piece-'u' pattern. Labellers anchor on the top-token list.
