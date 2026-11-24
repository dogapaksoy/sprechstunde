# sprechstunde

**A local assistant for official German letters that only says what it can prove.**

People who are new to Germany get letters from the Finanzamt, the Krankenkasse, the
Ausländerbehörde, the Jobcenter, the Beitragsservice. Missing a deadline in one of them is
expensive. `sprechstunde` answers questions about these letters ("Bis wann muss ich die
Kontoauszüge einreichen?") and sets reminders. Every date and amount it says is checked
against the letter first, and if the check fails it says so instead of guessing.

It runs fully offline on a laptop CPU with small open models (Qwen3 1.7B, escalating to
Qwen3 4B), because these letters should not leave the user's machine.

The answer is **spoken** (Whisper in, Piper out, both local). In a voice interface the
user cannot see a source link or skim the letter. What is said is all they get, so a
confident wrong date is the worst possible failure.

## Results

60 tasks on 12 synthetic mailboxes with OCR noise (36 answerable questions, 12 reminder
requests, 12 unanswerable questions; [setup below](#benchmark)). Laptop CPU, 4 threads.

| condition | answers correct | **answers wrong** | gave up | unanswerable: **hallucinated** | reminders correct | escalated | s / task (median, p90) |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B, no checks | 64% | **14%** | 22% | **50%** | 25% | – | 20, 26 |
| Qwen3-4B, no checks | 78% | **14%** | 8% | **58%** | 50% | – | 33, 50 |
| 1.7B + grounding | 58% | **8%** | 33% | **50%** | 42% | – | 20, 35 |
| 1.7B + grounding + topic | 56% | **3%** | 42% | **0%** | 25% | – | 22, 33 |
| 1.7B + grounding + escalation to 4B | 72% | **8%** | 19% | **50%** | 58% | 32% | 23, 86 |
| **1.7B + grounding + topic + escalation to 4B** | **72%** | **3%** | 25% | **0%** | 50% | 42% | 27, 89 |

- **The larger model alone does not fix it.** Qwen3-4B answers more questions, but it is
  just as often wrong (14%) and invents an answer to an unanswerable question even more
  often (58%).
- **Grounding halves wrong answers but does not stop hallucination.** Half of the
  unanswerable questions were still "answered" with real values from the wrong letter: the
  Finanzamt deadline for the *Hundesteuer*, the Ausländerbehörde fee for a *Reisepass*.
- **The topic check takes hallucinations from 50% to 0%** (12 of 12 blocked) and wrong
  answers from 8% to 3%. The one remaining wrong answer is an OCR error *in the letter*:
  the scan reads "8,00 €" where the original said "98,00 €". The answer is faithful to the
  text, so grounding can't catch it.
- **Escalation buys coverage back.** The 4B model is called for 42% of questions and lifts
  correct answers from 56% to 72%. Those questions take 50–100 s.
- With all checks, **96% of what is said is correct** (26 correct vs. 1 wrong value among
  spoken answers) and the assistant gave up on 25% of answerable questions. For a voice
  assistant about deadlines that trade is the point: "I can't confirm that" is recoverable,
  a wrong deadline is not.

The sample is small (36 answerable questions), so read differences of one or two tasks
(3–6 points) as noise. The large effects (hallucination 50% → 0%) are not.

## How it works

```
question ──► agent loop (Qwen3-1.7B, JSON actions, grammar-constrained)
               │ search_letters · read_letter · add_reminder · answer
               ▼
        answer {say, claims:[{type, value, source: "S12"}]}
               │
               ▼  guardrails, before anything is said or done
     1. grounding  every date/amount in `say` has a claim whose value belegt finds in
                   the cited sentence (OCR-tolerant)
     2. scope      if the question names a sender ("beim Jobcenter"), evidence must
                   come from that letter
     3. topic      the cited letter must be about the question's topics (NLI)
               │
     problems? ─► feed back to the model, 1 retry ─► escalate to Qwen3-4B
               │                                       └─► still failing: say
               ▼                                           "Das kann ich in Ihren Briefen
             speak                                          leider nicht sicher bestätigen."
```

Design decisions that mattered (all found by reading failing traces):

- **Cite sentence ids, don't copy quotes.** Search results are numbered (`S12`) and claims
  point at a number. This removed copy errors and cut the tokens a 1.7B model on a CPU has
  to generate per answer, which dominates latency.
- **Check what is said, not what is claimed.** Only claims whose value is actually spoken
  are verified, and every spoken date or amount needs a verified claim. A stray extra claim
  no longer blocks a correct answer, and a number without a claim can't sneak through.
- **Side effects are verified before they happen.** `add_reminder` only runs if its date is
  in the cited sentence and on topic, and then only after the user confirms. If the user
  asked for a reminder, an answer without one is rejected.
- **Grounding is not relevance.** belegt proves a value is in the cited sentence. It
  can't tell whether that sentence answers the question. Asked about the dog tax
  (*Hundesteuer*), which none of the letters mention, the model read out the real
  Finanzamt deadline, and grounding passed it. The topic check fixes this: German nouns are
  capitalised, so the question's topics are its specific capitalised nouns. Each one must
  be entailed (`Es geht um Hundesteuer.`) by the cited letter's sender, subject and cited
  sentences, judged by a multilingual NLI model. It gets world knowledge for free: a letter
  from the AOK about a *Beitragsrückstand* entails *Krankenkasse*, although the word never
  appears in it.
- **Small model first, large model only when needed.** Most questions are answered by the
  1.7B model. Only answers that fail the checks go to the 4B model.

## Voice

`sprechstunde ask letters/ question.wav --out answer.wav` transcribes the question with
Whisper large-v3-turbo, runs the agent, and speaks the answer with Piper
(`de_DE-thorsten-medium`). Both run on the CPU.

Verifying the text is not enough when the user only *hears* it. So before an answer is
played:

1. `speakable()` rewrites dates and amounts into spoken forms: "27.03.2026" →
   "27. März 2026", "4.799,10 €" → "4799,10 Euro".
2. **Readback check:** the synthesised audio is transcribed again, and the dates and
   amounts Whisper hears must equal the verified ones. If not, it is synthesised once more,
   slower. If that fails too, the assistant says it will show the answer on screen instead
   of playing a possibly wrong number.

The readback check is only as good as the ear. Piper's phonemes for numbers were right
("acht und zwanzig"), but Whisper wrote German compound numbers literally ("8 und 20",
"2006 und 20"), so correct audio looked wrong. A prompt showing numbers written as digits
and a small `repair_numbers` step fixed that.

`sprechstunde voice-eval` measures the voice layer without a microphone: Piper speaks,
Whisper listens.

**Answers** (100 distinct answers with a date or amount from all runs): does the user
hear the verified values?

| synthesised from | heard as | values match |
|---|---|---|
| raw answer text | plain Whisper | 68% |
| `speakable()` text | plain Whisper | 96% |
| `speakable()` text | number prompt + `repair_numbers` | 99% |
| full `say()` with one slower retry | | 100% |

`speakable()` is the big step (68% → 96%). Piper samples its audio, so the same text does
not sound the same twice, and the numbers vary by a point or two between runs (an earlier
run: 63%, 98%, 100%, and 2 of 92 answers fell back to the safe sentence).

**Questions** (22 distinct benchmark questions). Whisper first heard **"Jobcenter" as
"Jobcounter"**, which cost every task lost to speech (below). Questions are now
transcribed with a fixed list of German authority names as Whisper hotwords
(`voice.AUTHORITIES`, not tuned to the benchmark questions):

| audio | word error rate | topic words heard correctly | without hotwords |
|---|---|---|---|
| clean | 1.7% | 95% | 3.6%, 86% |
| white noise, SNR 10 dB | 2.9% | 95% | 6.1%, 86% |
| white noise, SNR 5 dB | 17.0% | 59% | 16.7%, 59% |

The remaining clean-audio miss is "Kontoauszüge" heard as "Kontorauszüge". Biasing Whisper
with the words from the user's own letters instead raised topic recall a little but made
it "hear" letter words in unanswerable questions, so it was not adopted.

**End to end:** the full agent (grounding + topic check + escalation) on the 60 tasks,
with every question spoken by Piper and transcribed by Whisper (clean audio):

| condition | answers correct | **answers wrong** | gave up | unanswerable: **hallucinated** | reminders correct | s / task (median, p90) |
|---|---|---|---|---|---|---|
| typed questions | 72% | **3%** | 25% | **0%** | 50% | 27, 89 |
| spoken, no hotwords | 64% | **3%** | 33% | **0%** | 42% | 30, 92 |
| **spoken, authority hotwords** | **72%** | **3%** | 25% | **0%** | 50% | 24, 86 |

Without hotwords, all 4 lost tasks were "Jobcounter" questions: the topic check found no
letter about a "Jobcounter", and the assistant said it could not find the answer. No
mishearing produced a wrong value; the guardrails turn speech errors into "I don't know",
not into wrong deadlines. With hotwords, spoken questions score the same as typed ones,
task for task (all 60 outcomes equal), although 16 of the 60 transcripts still differ from
the typed text ("Kontorauszüge", "musste" for "muss", a missing full stop).

Median CPU seconds: question transcription 4.6, answer synthesis 0.2, readback check 4.2.

## Benchmark

`sprechstunde eval` builds 12 synthetic mailboxes. Each has five letters (one per
authority) generated by [belegt](https://github.com/dogapaksoy/belegt) with 3% simulated OCR
noise and neutral ids (`brief1`–`brief5`), so the sender has to be read from the noisy
text. Each mailbox gets five tasks:

- three questions the letters answer
- one reminder request
- one question they **cannot** answer. These are chosen to tempt a wrong answer from a
  similar letter: *Hundesteuer* next to a Finanzamt letter, *Reisepass-Gebühr* next to an
  Ausländerbehörde fee, *Termin beim Bürgeramt* next to a Jobcenter appointment.

Scoring looks at the spoken answer: correct (gold value said), wrong (another value said),
gave up. An unanswerable question counts as hallucinated if any date or amount is said.
Every raw output, with every agent step, is in [`results/runs/`](results/runs).

## Usage

```bash
pip install "sprechstunde[local,nli,voice] @ git+https://github.com/dogapaksoy/sprechstunde.git"
sprechstunde chat path/to/letters/   # one .txt per letter; models download on first use
sprechstunde ask examples/letters/ question.wav --out answer.wav   # spoken question and answer
sprechstunde eval -n 12              # the benchmark above
sprechstunde voice-eval              # the voice benchmark
```

`--trace` prints every agent step. `--no-relevance` skips the NLI topic check if you don't
want to install torch.

## Limitations

- **Reminders are the weak spot.** Even with escalation only half are created correctly.
  The small models often answer the deadline instead of calling `add_reminder` and then run
  out of steps. The guard makes this fail safely (no false "Erinnerung angelegt"), but it
  doesn't make it succeed.
- **The topic check can block correct answers when OCR damages the header.** In one case
  the sender line read "Jobcente Berlin" and the subject line was unreadable, so the
  evidence did not entail "Jobcenter" (0.44 < 0.5) and the assistant gave up. The threshold
  was not tuned on the benchmark.
- **OCR errors in the letter itself pass through** (see "8,00 €" above). Detecting them
  needs a second signal, e.g. a plausibility check against the other amounts in the letter
  or asking the user to look at the line.
- **Latency.** On a CPU a question takes 20–30 s with the small model and up to ~90 s when
  it escalates, plus about 10 s for speech in and readback out. A voice interface needs a
  spoken "einen Moment, ich schaue nach" and a GPU or smaller prompts.
- **Synthetic speech.** The voice benchmark uses one synthetic voice for questions and
  white noise. Real users have accents, dialects, room echo and hesitations; this
  measures the pipeline, not robustness to real speech. At 5 dB SNR only 59% of the
  topic words survive.
- **Synthetic letters.** Five templates with simulated OCR noise, not real scans. Real
  letters have tables, multiple deadlines and legal boilerplate.
- Not legal advice. The assistant reads letters, it does not interpret them.

## License

MIT
