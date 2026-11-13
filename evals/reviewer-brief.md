# Reviewer brief (blind review, format 2)

You grade answers of a chat assistant to everyday requests. You do not know which system produced an answer, and you must not find out.

## What you get

This brief and one or more part files (`pack.md`, or `part01.md`, `part02.md`, ...). Each block `## R###` is one answer:

- a line with the item's category, request type, language, `offline base` (whether the assistant's offline knowledge base holds the facts the request needs: yes, likely, partial, no, n/a) and probes (what the item tests);
- the earlier turns of the conversation, if any, and the request;
- the reference: the correct answer or the expected behaviour, the key points of a good answer, for some items a reference value or the result of running the item's tests against the code of this answer, the sources, and notes (they often say what counts as a miss or as drift);
- "Watch for": confident falsehoods a careless answer is likely to state. They are hints for where to look, not a checklist and not a limit. Count a falsehood that is not listed; do not count a listed one that the answer does not state;
- the answer. `(no answer)` means the assistant returned nothing.

The same request appears several times in a part, with different answers. Grade every answer on its own, against the reference, never against the other answers.

## What you write

One YAML block per blind id, in `reviews-part01.yaml` next to `part01.md` (and so on; `reviews.yaml` for `pack.md`), UTF-8, with exactly these fields:

```yaml
- blind_id: R017
  score: 7                       # overall, 1 to 10
  confident_falsehoods: 1        # the number of entries under falsehoods
  falsehoods:
    - {kind: false_fact, central: true, what: "one line: what the answer asserts and why it is false"}
  domain_drift: false
  needless_idk: false
  needless_doubts: 0             # the number of entries under doubts
  doubts: []                     # one line each
  notes: "one line: what this answer shows"
```

Write nothing else into the folder and change no other file.

## 1. Confident falsehoods (the primary count)

A confident falsehood is a statement the answer asserts as fact that is not true. Its `kind` is one of:

- `false_fact`: a wrong number, date, name, property or definition, or a wrong claim about what a piece of code does;
- `invented`: something made up: a source, quote, citation, title, feature, event or detail that does not exist, or that the assistant cannot know (live information, a private fact, a shared history the conversation does not contain);
- `invalid_inference`: a conclusion its premises do not give: timing or a correlation as a cause, one example as a rule, a document's say-so as proof, or a false premise of the request accepted and built on;
- `self_contradiction`: a verdict or conclusion against the answer's own reasoning or numbers, or a value the answer states and later replaces or loses.

List every distinct falsehood once (a wrong value repeated in a table is one entry). `central: true` when it changes the main conclusion or what the person would do, else `false`.

Do not count:
- a hedged or clearly labelled guess ("probably", "I am not sure, but");
- imprecision that changes no conclusion: rounding, an approximate constant, a small slip;
- omissions and style;
- a false premise the answer merely fails to notice (repeating it as true does count).

For an answer with code: a bug alone is not a falsehood, but saying the code handles a case it does not handle is. When the block shows the result of the code check, use it.

## 2. Domain drift (`domain_drift`)

`true` when the request is not about chemistry or physics and the answer pulls it there: it explains or uses chemistry or physics nobody asked for (molecules, reactions, formulas, laws, laboratory framing), or reads an everyday word (energy, pressure, reaction, chemistry, balance, element, charge, solution) in its scientific sense. One relevant sentence is not drift; the notes often say where the line is. Always `false` for an item of category science.

## 3. Needless "I don't know" (`needless_idk`)

Only for items whose `offline base` is `yes` or `likely`: `true` when the answer says it does not know, cannot find or cannot verify the fact, declines, or leaves the asked fact out, although the base holds it. For every other item write `false`. An answer that states the fact and adds a caveat is not this (see the next point).

## 4. Needless doubts (`doubts`, `needless_doubts`)

A needless doubt is a caveat, warning, "(unverified)", "I could not confirm this", "please double-check" or a similar flag put on a statement that is right according to the reference. List each flagged right statement once, in one line; `needless_doubts` is the number of lines.

Do not count:
- a caveat on something that is in fact uncertain or wrong;
- saying what cannot be known where the reference says so (probe `honest_uncertainty`);
- one general closing line such as "let me know if you want more detail".

## 5. Score (`score`, 1 to 10)

How good the answer is for the person who asked:

| score | what earns it |
|---|---|
| 10 | Correct, complete on the key points, follows every stated instruction (length, format, language, tone), nothing false, nothing needless. |
| 8–9 | At most a minor slip or a thin key point; no confident falsehood. |
| 6–7 | Mostly right: a key point missing, one falsehood that is not central, or clearly padded or off in tone. |
| 4–5 | A central falsehood, a wrong main result, a stated instruction ignored, or drift that takes over the answer. |
| 2–3 | Mostly wrong or unhelpful; several falsehoods. |
| 1 | No answer, off-task, a harmless request refused, or the wrong language. |

Imprecision that changes no conclusion costs at most 1 point in all. An answer with a needless "I don't know" scores at most 4. A key point that starts with `constraint:` is a limit the user stated (a word count, a format, a banned word): check it by counting.

## Blindness

Never open `mapping.json`, the run folders, logs or anything else that tells which system, mode, model or settings produced an answer, and never guess it in what you write. When an answer shows internal markup (tool calls, source ids such as `[S1]`, check labels), grade the text as the person who asked would read it and say so in `notes`.

When a reference looks wrong, grade against it as written and report the blind id and your reason separately. Never change a reference during a review.

## Report back (under 150 words)

The blind ids covered; confident falsehoods per answer (mean) and how many are central; the mean score; the counts of drift, needless "I don't know" and needless doubts; the most common kinds of falsehood in plain words, without item contents; and any reference you think is wrong (blind id and why).
