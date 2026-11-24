### Questions: Piper -> Whisper large-v3-turbo

| audio | word error rate | topic words heard |
|---|---|---|
| clean | 3.6% | 86% (22 questions) |
| white noise, SNR 10 dB | 6.1% | 86% (22 questions) |
| white noise, SNR 5 dB | 16.7% | 59% (22 questions) |

### Answers: does the user hear the verified values? (92 distinct answers)

| synthesised from | heard as | values match |
|---|---|---|
| raw answer text | plain Whisper | 63% |
| `speakable()` text | plain Whisper | 98% |
| `speakable()` text | number prompt + `repair_numbers` | 100% |
| + one slower retry | | 98%; the rest falls back to a safe sentence |

### End to end: the full agent on spoken questions

| condition | answers correct | **answers wrong** | gave up | unanswerable: **hallucinated** | reminders correct | escalated | s / task (median, p90) |
|---|---|---|---|---|---|---|---|
| typed questions | 72% | **3%** | 25% | **0%** | 50% | 42% | 27, 89 |
| spoken questions (clean audio) | 64% | **3%** | 33% | **0%** | 42% | 40% | 30, 92 |

Median seconds on the CPU: question transcription 5.5, answer synthesis 0.3, readback check 5.2.
