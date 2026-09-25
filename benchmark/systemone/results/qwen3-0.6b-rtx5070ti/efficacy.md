## Label-free component selection

| component                 | mean ECE delta | worst acc delta | kept |
| ------------------------- | -------------- | --------------- | ---- |
| noul_case_variants=True   | -0.0068        | +0.0000         | yes  |
| noul_orders=2             | -0.0379        | -0.0100         | yes  |
| choice_name_variants=True | +0.0034        | -0.0133         | no   |
| choice_rotations=2        | -0.0727        | -0.0100         | no   |
| choice_rotations=4        | -0.1379        | -0.0300         | no   |
| choice_rotations=8        | -0.1356        | -0.0300         | no   |
| batch_prior=0.5           | -0.0567        | -0.0200         | no   |
| batch_prior=0.75          | -0.1028        | -0.0300         | no   |
| batch_prior=1.0           | -0.1576        | -0.0367         | no   |
| content_free=0.5          | -0.0337        | -0.0433         | no   |
| content_free=1.0          | -0.1061        | -0.0567         | no   |

Chosen: `{"noul_case_variants": true, "choice_name_variants": false, "noul_orders": 2, "choice_rotations": 1, "batch_prior": 0.0, "content_free": 0.0}`

## Reads

| task           | labels-only mass | mass with variants | reads disagree | reads |
| -------------- | ---------------- | ------------------ | -------------- | ----- |
| boolq          | 0.017            | 0.959              | 0.188          | 2     |
| sst2           | 0.559            | 0.998              | 0.068          | 2     |
| ag_news_sports | 0.718            | 0.998              | 0.045          | 2     |
| ag_news        | 0.795            | 0.795              | 0.335          | 4     |
| emotion        | 0.385            | 0.385              | 0.727          | 6     |
| banking77      | 0.161            | 0.161              | 1.000          | 8     |
| sst5           | 1.000            | 1.000              | 0.000          | 1     |
| yelp           | 0.999            | 0.999              | 0.000          | 1     |

## Metrics on the test half

| task           | kind   | strategy              | acc   | ECE   | NLL   | Brier | sel.acc@50% | decidable@5% | extra                         |
| -------------- | ------ | --------------------- | ----- | ----- | ----- | ----- | ----------- | ------------ | ----------------------------- |
| boolq          | noul   | raw                   | 0.663 | 0.283 | 1.215 | 0.583 | 0.840       | 0.017        | AUROC 0.731                   |
| boolq          | noul   | raw+fitted            | 0.690 | 0.082 | 0.578 | 0.394 | 0.840       | 0.017        | AUROC 0.731                   |
| boolq          | noul   | label_free            | 0.667 | 0.230 | 0.839 | 0.487 | 0.847       | 0.010        | AUROC 0.744                   |
| boolq          | noul   | label_free+fitted     | 0.690 | 0.080 | 0.569 | 0.386 | 0.833       | 0.010        | AUROC 0.744                   |
| boolq          | noul   | case variants only    | 0.663 | 0.262 | 1.028 | 0.538 | 0.840       | 0.010        | AUROC 0.732                   |
| boolq          | noul   | reads=2 only          | 0.673 | 0.238 | 0.910 | 0.511 | 0.860       | 0.023        | AUROC 0.739                   |
| boolq          | noul   | batch_prior=1.0 only  | 0.663 | 0.283 | 1.215 | 0.583 | 0.840       | 0.017        | AUROC 0.731                   |
| boolq          | noul   | content_free=1.0 only | 0.673 | 0.224 | 0.886 | 0.504 | 0.780       | 0.040        | AUROC 0.703                   |
| sst2           | noul   | raw                   | 0.520 | 0.438 | 1.624 | 0.857 | 0.733       | 0.017        | AUROC 0.755                   |
| sst2           | noul   | raw+fitted            | 0.720 | 0.085 | 0.592 | 0.403 | 0.767       | 0.020        | AUROC 0.755                   |
| sst2           | noul   | label_free            | 0.517 | 0.391 | 1.246 | 0.759 | 0.727       | 0.090        | AUROC 0.771                   |
| sst2           | noul   | label_free+fitted     | 0.707 | 0.078 | 0.579 | 0.391 | 0.773       | 0.013        | AUROC 0.771                   |
| sst2           | noul   | case variants only    | 0.520 | 0.432 | 1.565 | 0.846 | 0.740       | 0.017        | AUROC 0.759                   |
| sst2           | noul   | reads=2 only          | 0.523 | 0.395 | 1.270 | 0.766 | 0.727       | 0.073        | AUROC 0.769                   |
| sst2           | noul   | batch_prior=1.0 only  | 0.637 | 0.178 | 0.675 | 0.468 | 0.753       | 0.073        | AUROC 0.756                   |
| sst2           | noul   | content_free=1.0 only | 0.580 | 0.216 | 0.724 | 0.503 | 0.733       | 0.017        | AUROC 0.755                   |
| ag_news_sports | noul   | raw                   | 0.983 | 0.068 | 0.125 | 0.044 | 0.973       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | raw+fitted            | 0.970 | 0.030 | 0.080 | 0.045 | 1.000       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | label_free            | 0.973 | 0.033 | 0.080 | 0.039 | 1.000       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | label_free+fitted     | 0.973 | 0.033 | 0.080 | 0.039 | 1.000       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | case variants only    | 0.983 | 0.074 | 0.130 | 0.046 | 0.973       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | reads=2 only          | 0.973 | 0.034 | 0.081 | 0.040 | 1.000       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | batch_prior=1.0 only  | 0.977 | 0.110 | 0.185 | 0.068 | 0.967       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | content_free=1.0 only | 0.983 | 0.056 | 0.112 | 0.041 | 0.973       | 1.000        | AUROC 0.994                   |
| ag_news        | choice | raw                   | 0.840 | 0.124 | 0.788 | 0.275 | 0.920       | 0.400        |                               |
| ag_news        | choice | raw+fitted            | 0.840 | 0.045 | 0.477 | 0.252 | 0.927       | 0.357        |                               |
| ag_news        | choice | label_free            | 0.840 | 0.124 | 0.788 | 0.275 | 0.920       | 0.400        |                               |
| ag_news        | choice | label_free+fitted     | 0.840 | 0.045 | 0.477 | 0.252 | 0.927       | 0.357        |                               |
| ag_news        | choice | name variants only    | 0.840 | 0.124 | 0.788 | 0.275 | 0.920       | 0.400        |                               |
| ag_news        | choice | reads=8 only          | 0.810 | 0.124 | 0.768 | 0.308 | 0.940       | 0.333        |                               |
| ag_news        | choice | batch_prior=1.0 only  | 0.850 | 0.110 | 0.739 | 0.258 | 0.947       | 0.483        |                               |
| ag_news        | choice | content_free=1.0 only | 0.783 | 0.161 | 1.179 | 0.385 | 0.893       | 0.267        |                               |
| emotion        | choice | raw                   | 0.500 | 0.275 | 2.303 | 0.798 | 0.607       | 0.003        |                               |
| emotion        | choice | raw+fitted            | 0.500 | 0.085 | 1.440 | 0.682 | 0.600       | 0.000        |                               |
| emotion        | choice | label_free            | 0.500 | 0.275 | 2.303 | 0.798 | 0.607       | 0.003        |                               |
| emotion        | choice | label_free+fitted     | 0.500 | 0.085 | 1.440 | 0.682 | 0.600       | 0.000        |                               |
| emotion        | choice | name variants only    | 0.487 | 0.283 | 2.303 | 0.798 | 0.607       | 0.003        |                               |
| emotion        | choice | reads=8 only          | 0.527 | 0.269 | 1.828 | 0.735 | 0.633       | 0.000        |                               |
| emotion        | choice | batch_prior=1.0 only  | 0.463 | 0.295 | 2.252 | 0.824 | 0.593       | 0.010        |                               |
| emotion        | choice | content_free=1.0 only | 0.493 | 0.275 | 1.860 | 0.757 | 0.640       | 0.020        |                               |
| banking77      | choice | raw                   | 0.073 | 0.485 | 8.200 | 1.277 | 0.120       | 0.033        |                               |
| banking77      | choice | raw+fitted            | 0.073 | 0.034 | 4.258 | 0.982 | 0.113       | 0.030        |                               |
| banking77      | choice | label_free            | 0.073 | 0.485 | 8.200 | 1.277 | 0.120       | 0.033        |                               |
| banking77      | choice | label_free+fitted     | 0.073 | 0.034 | 4.258 | 0.982 | 0.113       | 0.030        |                               |
| banking77      | choice | name variants only    | 0.070 | 0.488 | 8.165 | 1.278 | 0.113       | 0.033        |                               |
| banking77      | choice | reads=8 only          | 0.263 | 0.085 | 3.517 | 0.853 | 0.480       | 0.007        |                               |
| banking77      | choice | batch_prior=1.0 only  | 0.093 | 0.104 | 4.694 | 0.980 | 0.160       | 0.023        |                               |
| banking77      | choice | content_free=1.0 only | 0.090 | 0.189 | 4.897 | 1.005 | 0.167       | 0.020        |                               |
| sst5           | score  | raw                   | 0.220 | 0.766 | 5.092 | 1.530 | 0.280       | 0.007        | MAE 1.278 RPS 0.316 QWK 0.000 |
| sst5           | score  | raw+fitted            | 0.220 | 0.063 | 1.604 | 0.798 | 0.307       | 0.007        | MAE 1.148 RPS 0.187 QWK 0.000 |
| sst5           | score  | label_free            | 0.220 | 0.766 | 5.092 | 1.530 | 0.280       | 0.007        | MAE 1.278 RPS 0.316 QWK 0.000 |
| sst5           | score  | label_free+fitted     | 0.220 | 0.063 | 1.604 | 0.798 | 0.307       | 0.007        | MAE 1.148 RPS 0.187 QWK 0.000 |
| sst5           | score  | batch_prior=1.0 only  | 0.237 | 0.230 | 1.703 | 0.859 | 0.267       | 0.000        | MAE 1.040 RPS 0.180 QWK 0.265 |
| sst5           | score  | content_free=1.0 only | 0.270 | 0.202 | 1.808 | 0.849 | 0.300       | 0.003        | MAE 1.073 RPS 0.181 QWK 0.230 |
| yelp           | score  | raw                   | 0.247 | 0.265 | 1.994 | 0.919 | 0.267       | 0.017        | MAE 1.141 RPS 0.207 QWK 0.144 |
| yelp           | score  | raw+fitted            | 0.247 | 0.035 | 1.578 | 0.788 | 0.300       | 0.003        | MAE 1.164 RPS 0.189 QWK 0.144 |
| yelp           | score  | label_free            | 0.247 | 0.265 | 1.994 | 0.919 | 0.267       | 0.017        | MAE 1.141 RPS 0.207 QWK 0.144 |
| yelp           | score  | label_free+fitted     | 0.247 | 0.035 | 1.578 | 0.788 | 0.300       | 0.003        | MAE 1.164 RPS 0.189 QWK 0.144 |
| yelp           | score  | batch_prior=1.0 only  | 0.310 | 0.118 | 1.581 | 0.789 | 0.353       | 0.007        | MAE 1.088 RPS 0.184 QWK 0.311 |
| yelp           | score  | content_free=1.0 only | 0.210 | 0.500 | 2.653 | 1.125 | 0.240       | 0.000        | MAE 1.274 RPS 0.265 QWK 0.072 |
