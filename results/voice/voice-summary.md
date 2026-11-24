### Questions: Piper -> Whisper large-v3-turbo with authority hotwords

| audio | word error rate | topic words heard |
|---|---|---|
| clean | 1.7% | 95% (22 questions) |
| white noise, SNR 10 dB | 2.9% | 95% (22 questions) |
| white noise, SNR 5 dB | 17.0% | 59% (22 questions) |

### Answers: does the user hear the verified values? (100 distinct answers)

| synthesised from | heard as | values match |
|---|---|---|
| raw answer text | plain Whisper | 68% |
| `speakable()` text | plain Whisper | 96% |
| `speakable()` text | number prompt + `repair_numbers` | 99% |
| + one slower retry | | 100%; the rest falls back to a safe sentence |

### End to end: the full agent on spoken questions

| condition | answers correct | **answers wrong** | gave up | unanswerable: **hallucinated** | reminders correct | escalated | s / task (median, p90) |
|---|---|---|---|---|---|---|---|
| typed questions | 72% | **3%** | 25% | **0%** | 50% | 42% | 27, 89 |
| spoken questions (clean audio) | 72% | **3%** | 25% | **0%** | 50% | 42% | 24, 86 |

Median seconds on the CPU: question transcription 4.6, answer synthesis 0.2, readback check 4.2.
