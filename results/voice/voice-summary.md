### Questions (Piper -> Whisper small)

| audio | word error rate | topics kept |
|---|---|---|
| clean | 6.3% | 73% (22 questions) |
| white noise, SNR 10 dB | 21.8% | 45% (22 questions) |
| white noise, SNR 5 dB | 41.7% | 23% (22 questions) |

### Answers with a date or amount (89 distinct answers)

| what is synthesised | user hears the verified values |
|---|---|
| raw answer text | 35% |
| `speakable()` text | 60% |
| `speakable()` + readback check (else fallback) | 100% by construction; fallback on 22% |

Median seconds on the CPU: question transcription 1.42, answer synthesis 0.26, readback check 1.61.
