## Label-free component selection

| component                 | mean ECE delta | worst acc delta | kept |
| ------------------------- | -------------- | --------------- | ---- |
| noul_case_variants=True   | -0.0507        | +0.0300         | yes  |
| noul_orders=2             | -0.0359        | +0.0400         | yes  |
| choice_name_variants=True | +0.0076        | -0.0033         | no   |
| choice_rotations=2        | +0.0002        | +0.0200         | no   |
| choice_rotations=4        | +0.0126        | +0.0167         | no   |
| choice_rotations=8        | +0.0079        | +0.0167         | no   |
| batch_prior=0.5           | -0.0094        | -0.0167         | no   |
| batch_prior=0.75          | -0.0096        | -0.0200         | no   |
| batch_prior=1.0           | -0.0129        | -0.0300         | no   |
| content_free=0.5          | +0.0307        | -0.0567         | no   |
| content_free=1.0          | +0.0833        | -0.1233         | no   |

Chosen: `{"noul_case_variants": true, "choice_name_variants": false, "noul_orders": 2, "choice_rotations": 1, "batch_prior": 0.0, "content_free": 0.0}`

## Reads

| task           | labels-only mass | mass with variants | reads disagree | reads |
| -------------- | ---------------- | ------------------ | -------------- | ----- |
| boolq          | 0.006            | 0.925              | 0.340          | 2     |
| sst2           | 0.063            | 0.707              | 0.173          | 2     |
| ag_news_sports | 0.059            | 0.912              | 0.158          | 2     |
| ag_news        | 0.202            | 0.202              | 0.205          | 4     |
| emotion        | 0.032            | 0.037              | 0.542          | 6     |
| banking77      | 0.118            | 0.120              | 0.712          | 8     |
| sst5           | 0.336            | 0.336              | 0.000          | 1     |
| yelp           | 0.143            | 0.143              | 0.000          | 1     |

## Metrics on the test half

| task           | kind   | strategy              | acc   | ECE   | NLL   | Brier | sel.acc@50% | decidable@5% | extra                         |
| -------------- | ------ | --------------------- | ----- | ----- | ----- | ----- | ----------- | ------------ | ----------------------------- |
| boolq          | noul   | raw                   | 0.663 | 0.277 | 0.676 | 0.451 | 0.760       | 0.003        | AUROC 0.873                   |
| boolq          | noul   | raw+fitted            | 0.803 | 0.073 | 0.429 | 0.271 | 0.920       | 0.200        | AUROC 0.873                   |
| boolq          | noul   | label_free            | 0.803 | 0.138 | 0.469 | 0.296 | 0.900       | 0.000        | AUROC 0.891                   |
| boolq          | noul   | label_free+fitted     | 0.820 | 0.060 | 0.404 | 0.252 | 0.927       | 0.363        | AUROC 0.891                   |
| boolq          | noul   | case variants only    | 0.693 | 0.252 | 0.634 | 0.416 | 0.787       | 0.000        | AUROC 0.874                   |
| boolq          | noul   | reads=2 only          | 0.803 | 0.139 | 0.468 | 0.298 | 0.887       | 0.133        | AUROC 0.888                   |
| boolq          | noul   | batch_prior=1.0 only  | 0.663 | 0.277 | 0.676 | 0.451 | 0.760       | 0.003        | AUROC 0.873                   |
| boolq          | noul   | content_free=1.0 only | 0.737 | 0.064 | 0.502 | 0.335 | 0.893       | 0.237        | AUROC 0.806                   |
| sst2           | noul   | raw                   | 0.590 | 0.276 | 0.692 | 0.500 | 0.807       | 0.203        | AUROC 0.896                   |
| sst2           | noul   | raw+fitted            | 0.823 | 0.079 | 0.410 | 0.262 | 0.927       | 0.427        | AUROC 0.896                   |
| sst2           | noul   | label_free            | 0.803 | 0.156 | 0.483 | 0.311 | 0.920       | 0.323        | AUROC 0.906                   |
| sst2           | noul   | label_free+fitted     | 0.817 | 0.073 | 0.378 | 0.248 | 0.967       | 0.520        | AUROC 0.906                   |
| sst2           | noul   | case variants only    | 0.763 | 0.160 | 0.508 | 0.334 | 0.867       | 0.277        | AUROC 0.896                   |
| sst2           | noul   | reads=2 only          | 0.633 | 0.241 | 0.600 | 0.421 | 0.800       | 0.243        | AUROC 0.907                   |
| sst2           | noul   | batch_prior=1.0 only  | 0.797 | 0.147 | 0.494 | 0.319 | 0.893       | 0.343        | AUROC 0.896                   |
| sst2           | noul   | content_free=1.0 only | 0.797 | 0.147 | 0.495 | 0.320 | 0.900       | 0.327        | AUROC 0.896                   |
| ag_news_sports | noul   | raw                   | 0.857 | 0.142 | 0.258 | 0.177 | 1.000       | 0.787        | AUROC 0.992                   |
| ag_news_sports | noul   | raw+fitted            | 0.957 | 0.048 | 0.114 | 0.065 | 1.000       | 1.000        | AUROC 0.992                   |
| ag_news_sports | noul   | label_free            | 0.973 | 0.140 | 0.189 | 0.078 | 1.000       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | label_free+fitted     | 0.967 | 0.044 | 0.095 | 0.052 | 1.000       | 1.000        | AUROC 0.994                   |
| ag_news_sports | noul   | case variants only    | 0.920 | 0.130 | 0.200 | 0.109 | 1.000       | 0.933        | AUROC 0.993                   |
| ag_news_sports | noul   | reads=2 only          | 0.937 | 0.109 | 0.161 | 0.084 | 1.000       | 0.957        | AUROC 0.994                   |
| ag_news_sports | noul   | batch_prior=1.0 only  | 0.967 | 0.111 | 0.182 | 0.083 | 1.000       | 1.000        | AUROC 0.992                   |
| ag_news_sports | noul   | content_free=1.0 only | 0.950 | 0.107 | 0.175 | 0.084 | 1.000       | 1.000        | AUROC 0.992                   |
| ag_news        | choice | raw                   | 0.890 | 0.044 | 0.336 | 0.168 | 0.987       | 0.823        |                               |
| ag_news        | choice | raw+fitted            | 0.890 | 0.057 | 0.327 | 0.166 | 0.993       | 0.827        |                               |
| ag_news        | choice | label_free            | 0.890 | 0.044 | 0.336 | 0.168 | 0.987       | 0.823        |                               |
| ag_news        | choice | label_free+fitted     | 0.890 | 0.057 | 0.327 | 0.166 | 0.993       | 0.827        |                               |
| ag_news        | choice | name variants only    | 0.890 | 0.044 | 0.336 | 0.168 | 0.987       | 0.823        |                               |
| ag_news        | choice | reads=8 only          | 0.907 | 0.046 | 0.288 | 0.144 | 0.993       | 0.870        |                               |
| ag_news        | choice | batch_prior=1.0 only  | 0.897 | 0.052 | 0.321 | 0.161 | 0.987       | 0.833        |                               |
| ag_news        | choice | content_free=1.0 only | 0.840 | 0.078 | 0.582 | 0.272 | 0.947       | 0.470        |                               |
| emotion        | choice | raw                   | 0.460 | 0.200 | 1.709 | 0.746 | 0.613       | 0.000        |                               |
| emotion        | choice | raw+fitted            | 0.460 | 0.065 | 1.482 | 0.692 | 0.620       | 0.003        |                               |
| emotion        | choice | label_free            | 0.460 | 0.200 | 1.709 | 0.746 | 0.613       | 0.000        |                               |
| emotion        | choice | label_free+fitted     | 0.460 | 0.065 | 1.482 | 0.692 | 0.620       | 0.003        |                               |
| emotion        | choice | name variants only    | 0.467 | 0.220 | 1.684 | 0.728 | 0.640       | 0.000        |                               |
| emotion        | choice | reads=8 only          | 0.513 | 0.223 | 1.507 | 0.707 | 0.647       | 0.007        |                               |
| emotion        | choice | batch_prior=1.0 only  | 0.430 | 0.193 | 1.823 | 0.772 | 0.587       | 0.010        |                               |
| emotion        | choice | content_free=1.0 only | 0.357 | 0.310 | 2.215 | 0.892 | 0.493       | 0.013        |                               |
| banking77      | choice | raw                   | 0.600 | 0.098 | 2.108 | 0.579 | 0.787       | 0.003        |                               |
| banking77      | choice | raw+fitted            | 0.600 | 0.097 | 1.978 | 0.576 | 0.800       | 0.003        |                               |
| banking77      | choice | label_free            | 0.600 | 0.098 | 2.108 | 0.579 | 0.787       | 0.003        |                               |
| banking77      | choice | label_free+fitted     | 0.600 | 0.097 | 1.978 | 0.576 | 0.800       | 0.003        |                               |
| banking77      | choice | name variants only    | 0.597 | 0.101 | 2.108 | 0.581 | 0.793       | 0.003        |                               |
| banking77      | choice | reads=8 only          | 0.700 | 0.097 | 1.434 | 0.429 | 0.887       | 0.277        |                               |
| banking77      | choice | batch_prior=1.0 only  | 0.660 | 0.066 | 1.750 | 0.489 | 0.847       | 0.087        |                               |
| banking77      | choice | content_free=1.0 only | 0.643 | 0.123 | 2.033 | 0.545 | 0.787       | 0.217        |                               |
| sst5           | score  | raw                   | 0.273 | 0.262 | 1.805 | 0.887 | 0.333       | 0.000        | MAE 0.975 RPS 0.178 QWK 0.268 |
| sst5           | score  | raw+fitted            | 0.273 | 0.032 | 1.541 | 0.774 | 0.340       | 0.000        | MAE 1.062 RPS 0.169 QWK 0.268 |
| sst5           | score  | label_free            | 0.273 | 0.262 | 1.805 | 0.887 | 0.333       | 0.000        | MAE 0.975 RPS 0.178 QWK 0.268 |
| sst5           | score  | label_free+fitted     | 0.273 | 0.032 | 1.541 | 0.774 | 0.340       | 0.000        | MAE 1.062 RPS 0.169 QWK 0.268 |
| sst5           | score  | batch_prior=1.0 only  | 0.270 | 0.162 | 1.513 | 0.794 | 0.347       | 0.003        | MAE 0.909 RPS 0.150 QWK 0.391 |
| sst5           | score  | content_free=1.0 only | 0.250 | 0.509 | 2.508 | 1.145 | 0.347       | 0.000        | MAE 1.061 RPS 0.224 QWK 0.171 |
| yelp           | score  | raw                   | 0.277 | 0.178 | 1.790 | 0.856 | 0.267       | 0.007        | MAE 1.190 RPS 0.218 QWK 0.183 |
| yelp           | score  | raw+fitted            | 0.277 | 0.038 | 1.574 | 0.786 | 0.280       | 0.003        | MAE 1.180 RPS 0.193 QWK 0.183 |
| yelp           | score  | label_free            | 0.277 | 0.178 | 1.790 | 0.856 | 0.267       | 0.007        | MAE 1.190 RPS 0.218 QWK 0.183 |
| yelp           | score  | label_free+fitted     | 0.277 | 0.038 | 1.574 | 0.786 | 0.280       | 0.003        | MAE 1.180 RPS 0.193 QWK 0.183 |
| yelp           | score  | batch_prior=1.0 only  | 0.247 | 0.163 | 1.667 | 0.832 | 0.253       | 0.007        | MAE 1.155 RPS 0.200 QWK 0.236 |
| yelp           | score  | content_free=1.0 only | 0.220 | 0.253 | 2.054 | 0.911 | 0.240       | 0.010        | MAE 1.229 RPS 0.218 QWK 0.327 |
