Reasoning effort high, budget 4096 tokens, 50 test rows per task.

| task | one-pass acc | think-then-read acc | one-pass ECE | think-then-read ECE | one-pass NLL | think-then-read NLL | reasoning tokens (mean) | truncated | s/question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boolq | 0.660 | 0.860 | 0.362 | 0.140 | 0.812 | 1.228 | 568 | 0.04 | 8.92 |
| sst2 | 0.740 | 0.920 | 0.216 | 0.074 | 0.510 | 0.550 | 357 | 0.00 | 5.49 |
| ag_news_sports | 0.860 | 0.900 | 0.184 | 0.096 | 0.250 | 0.529 | 442 | 0.00 | 6.76 |
| ag_news | 0.880 | 0.860 | 0.133 | 0.140 | 0.374 | 1.754 | 325 | 0.00 | 5.05 |
| emotion | 0.460 | 0.600 | 0.201 | 0.401 | 1.685 | 5.379 | 669 | 0.04 | 10.26 |
| sst5 | 0.340 | 0.580 | 0.235 | 0.410 | 1.662 | 3.806 | 762 | 0.00 | 11.45 |
| yelp | 0.240 | 0.440 | 0.200 | 0.533 | 1.924 | 4.293 | 1283 | 0.04 | 19.69 |
