| condition | answers correct | **answers wrong** | gave up | unanswerable: **hallucinated** | reminders correct | escalated | s / task (median, p90) |
|---|---|---|---|---|---|---|---|
| qwen3-1.7b, no checks | 64% | **14%** | 22% | **50%** | 25% | 0% | 20, 26 |
| qwen3-1.7b + grounding | 58% | **8%** | 33% | **50%** | 42% | 0% | 20, 35 |
| qwen3-1.7b + grounding + topic | 56% | **3%** | 42% | **0%** | 25% | 0% | 22, 33 |
| qwen3-1.7b + grounding + escalation to qwen3-4b | 72% | **8%** | 19% | **50%** | 58% | 32% | 23, 86 |
| **qwen3-1.7b + grounding + topic + escalation to qwen3-4b** | 72% | **3%** | 25% | **0%** | 50% | 42% | 27, 89 |
| qwen3-4b, no checks | 78% | **14%** | 8% | **58%** | 50% | 0% | 33, 50 |
