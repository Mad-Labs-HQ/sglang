Reasoning effort medium, budget 2048 tokens, 50 test rows per task.

| task | one-pass acc | think-then-read acc | one-pass ECE | think-then-read ECE | one-pass NLL | think-then-read NLL | reasoning tokens (mean) | truncated | s/question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boolq | 0.680 | 0.880 | 0.355 | 0.119 | 0.806 | 0.778 | 128 | 0.00 | 2.38 |
| sst2 | 0.740 | 0.940 | 0.214 | 0.061 | 0.511 | 0.384 | 94 | 0.00 | 1.88 |
| ag_news_sports | 0.860 | 0.960 | 0.183 | 0.037 | 0.249 | 0.229 | 82 | 0.00 | 1.66 |
| ag_news | 0.880 | 0.840 | 0.136 | 0.160 | 0.380 | 1.756 | 50 | 0.00 | 1.21 |
| emotion | 0.460 | 0.620 | 0.228 | 0.380 | 1.697 | 4.550 | 113 | 0.00 | 2.13 |
| sst5 | 0.340 | 0.500 | 0.240 | 0.498 | 1.684 | 4.376 | 131 | 0.00 | 2.42 |
| yelp | 0.240 | 0.360 | 0.202 | 0.579 | 1.913 | 4.796 | 214 | 0.00 | 3.71 |
