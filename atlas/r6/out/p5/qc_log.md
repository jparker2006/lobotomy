# R6 planner p5 — detailed QC log (working notes; summary in qc.md)

## QC log (v1)
- b0 (orchestrator wave): read L26n1697, L26n1564, L26n2357. Labels match the evidence. L26n1564 "Digits inside
  numbers" = the 2nd+ digit of numbers (0<<0>>0 AD, 2,000), right. L26n2357 "Tech brand names" = mostly Google,
  plus Adobe/BlackBerry, writes "camera", right. L26n1697 "Vehicle and transport nouns" = boat/car/space before a
  noun in 6/15 snippets, rest are word fragments, so `likely` is a bit generous. 15/20 are `mixed` word-onset
  fragment poles (sp-, cr-, B-), typical for L26. No re-run. Dev r +0.354 (word list +0.296).
- b1: L26n981 "'the' before named entities" (the former Soviet Union, the Ottoman) and L26n999 "'hot' and food-list
  pieces" both match the evidence, and both are honestly `mixed`. OK.
- b2: L26n2358 "Preposition before place names" (in Dubai, between Johannesburg and Durban) is right, but its 10
  concepts are top-token place fragments (Warsaw, aj, Nan) that don't match the description. L25n1786 "Modal before
  action verb" (users can| install) is right, but its concepts are fragments (ogy, Rel). OK, no re-run.
- b3: L26n1591 "Stem before -age ending" (lever|ages, volt|ages, gar|ages; writes 'agent'), good. OK.
- b4: L26n268 "Acronym letters after parenthesis" (`clear`, (ATL), (AIDS), U|AB) and L26n455 "Negating un- prefix"
  both good. OK.
- b5: L26n1532 "Material or substance nouns" (compost or bark, foam, cumin) is a fair `likely`. OK.
- b6: L25n2834 "'the' before cause nouns" (the| cause of obesity, the| root) is a deserved `clear`. OK.
- b7: L26n759 "Infinitive 'to' and 'wh'" is weak. 7/15 peaks are 'to' after an actor (users to| browse, calls to|
  ask), but the name and concepts lead with wh/car, and 'to' is not among the concepts. Watch its dev r.
- b8: L25n1364 "Tech-stack link → node" (app using| React) is a good `likely`. L26n782 "Diffuse; writes capital M"
  is honestly diffuse. OK.
- Systemic: concepts often copy the top-token list (subword fragments) instead of the trigger the description names.
  That may cost scorer r on fragment poles. Not editable under the rules, so I'll look at it at the dev-score step.
- b9: L26n317 "Think-tag start, filler" fires on `<think>` in 15/15 snippets, so `mixed` under-claims (should be
  `clear`). Not editable; the label text is right. L25n2194 "Shell and sudo commands" is fine.
- b10: L26n2617 "Room and space sufficiency" (free| space, enough, ample, a lot; writes 'room'), a good `clear`.
- b11: L26n985 "Institution-name opener, economic" is `likely`, but its top snippets are fragments (D|ass|ault,
  som|atostatin, metaph|ysical). World Bank / American Journal only show up at act 5, so the label fits the
  weak tail. L26n1646/L26n828 concepts carry meta words ("split words", "closing parens"). **Flag b11.**
- b12: L25n2254 "hash / digest word", a good `clear`. L26n394 "SQL and code syntax keywords" (OFFSET, ORDER BY, =(),
  OK as `mixed`.
- b13: L25n2150 "Word before month name" is `likely`, but the top peaks are 'Ger'trude, 'ger'mline, 'convers'ational
  (4x), 'ger'iatrics. Only ~5/15 sit near a month or season, so the label leans on the writes (month names).
  L26n3038 "Color words" `clear` looks right. **Flag b13.**
- b14: L25n607 "Daily and threshold limits" (recommended| daily intake, threshold values) is a fair `likely`.
- b17: L26n1425 "Stem before '-ist'/'-ian'" (hydro|log|ist, mathematic|ian) is a deserved `clear`. L26n219
  "'mental' in wellbeing text" leads with 'mental', but 7/15 peaks are digits/characters inside URLs, DOIs and
  decimals (youtu.be/..R|uz|, doi.org/bnx|8|, 0.|6| mg), so it is half right.
- b19: L26n2283 "'Full' in view links" (view| full size, “Full Text”; writes 'text') is a fair `likely`.
- b22: L26n1959 "Gender and sexuality words" (limiting| gender stereotypes, sexually) is right as `mixed`.
- b23: L25n1782 "Earth-science terms (rock, soil)" (soil quality, rock mass, forest structure; writes 'property')
  is good, but its concepts are top-token junk ('=(', 'ALL', ']$').
- b24: L26n1480 "Treaty and signing words" `clear` is too strong: the peak is 'was' in "Agreement was| signed", the
  rest are weak (4–6). L25n1526 "Word before importance adjective" (why is it| important) is a good `likely`.
- b15: L26n2193 "Computer hardware and software" (network| card, GPU, USB; writes 'driver'): label right, `clear`
  a bit generous.
- b16: L26n2 "Answer-start and number tokens": the main pattern, the space before a number (as early as| 3640 BCE,
  oven to| 300°F), is right. Concepts include meta strings ("space before number", "The after 1.").
- b18: L25n1833 "Games/sports -> 'win'" (rugby| touch, playing tennis) is a fair `likely`.
- b20: L26n87 "Answer-opening 'Here's'" (</think> Here|'s a…, to put| it simply), right.
- b21: L26n3007 "'years' in age phrases" (thirteen| years old; writes 'old'), a deserved `clear`.
- b25: L26n765 "People-group nouns before possessive" (League of| Women, students|’), right.
- b26: L25n1274 "Closing think-tag token" fires on `</think>` in 15/15 snippets, so `mixed` under-claims again
  (like L26n317). Labellers mark single-token chat-template poles `mixed`, probably because the writes are junk.
- Names: shortened 57 names over 4 words, keeping their meaning (arrows → commas, "X and Y" → "X/Y", for example
  "Foreign name fragments → download" → "Foreign name fragments, download"). The old name is kept in `name_orig`.
- v1 totals: 542/542 labelled, 0 failed. Verdicts: clear 36, likely 110, mixed 396.

## Dev score v1 and re-run choice
- v1 dev (542 poles): real r 0.371, word list 0.187 (n=315), lookup 0.473. real−wordlist +0.164 [0.127, 0.198].
  By verdict: clear 0.581 (36), likely 0.430 (110), mixed 0.335 (396). No batch is under 0.10.
- Worst batches: b7 0.110, b22 0.207, b10 0.263, b4 0.293. Best: b24 0.529.
- b22 is partly a scorer failure. One Haiku scorer job gave constant predictions (all 1s, so r=0) for 7 poles
  (L25n1025, L25n1438, L25n2875, L26n57, L26n1210, L25n1982, L26n322). L25n2875 "'Public' and 'public domain'" is
  a plain label that should be easy to score.
- Re-run v2: b7, b22, b10 (worst 3), b11 and b13 (QC flags), plus b4, because b22's low score is partly the scorer.
- Opus second opinion (decides re-runs only): the 7 constant-scored poles, plus 3 labels that read right on the
  evidence but scored ≤ −0.05: L26n2617 "Room and space sufficiency" (−0.05), L25n517 "Festival and holiday words"
  (−0.14), L26n2595 "Name-start pieces" (−0.21).
- Opus result (dev r, Haiku → Opus): L25n2875 0→0.65, L26n1210 0→0.70, L25n1438 0→0.55, L26n57 0→0.54,
  L25n1982 0→0.53, L25n1025 0→0.34, L26n322 0→0.14, L26n2595 −0.21→0.36, L25n517 −0.14→0.26, L26n2617 −0.05→0.22.
  So all 10 labels carry signal, and their low scores were a confused or lazy Haiku scorer. Decision: no extra
  re-runs on their account. I kept b22 in v2 since it was already queued; the per-pole pick decides. The 7 flat
  predictions were rescored on dev with the same Haiku scorer (allowed on dev by orchestrator rule). The old flat
  predictions are kept in `scores_dev_real_v1_flat.json`. Opus scores never enter any final number.
- After the flat rescore, v1 dev real is 0.377 (b22 rises off the worst list).
- v2 (120 poles, b4/7/10/11/13/22): 120/120 labelled, 0 failed. I shortened 2 long names. The v2 labels fix the
  QC flags. L25n2150 is now "Months and conversational tone" (names the convers/ger peaks). L26n985 is now
  "Word-start stem pieces" (leads with som|atostatin and metaph|ysical). L26n759 is now "Word before bare verb"
  (leads with 'to'). Some v2 concepts carry "(word)" glosses such as "cr (crickets)".
- v2 dev (120): 0.385 overall. Per batch, v1 → v2: b7 0.110→0.340, b10 0.263→0.413, b22 0.359→0.445,
  b4 0.293→0.326, b13 0.329→0.332, b11 0.494→0.454. No v2 flat predictions. No batch is under 0.10, so no v3.
- Pilot dev (7): beats v1/v2 on 4 poles (L25n2845 0.68 vs 0.24, L25n2781 0.55 vs 0.22, L26n876, L26n2867).
  It loses badly on L25n1364 (−0.28 vs v1 0.84).
- Pick (v1,v2,pilot): 542 final labels = v1 458, v2 80, pilot 4. Verdicts: clear 33, likely 124, mixed 385.
  The final dev mean is 0.406, but that is best-of-versions, so it is biased up. The honest number is the test
  score below.
