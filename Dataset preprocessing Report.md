# Dataset preprocessing Report

- Files processed: **18**

## Sources

- Merged train rows: **16,748,125**
- Train sources: ['ids2017', 'ids2018']
- Observation features (17): ['Flow Pkts/s', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Flow Byts/s', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Active Conn Cnt', 'Flow Duration', 'Pkt Len Mean', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Dst Port Bucket', 'Protocol']
- Hidden (reward/eval only): ['attack_label', 'mdp_state', 'TimestampEpoch', 'source_day', 'source_dataset']
- Scaler fit on: merged train split only (ids2018 + ids2017; all holdout days excluded)

## Holdout splits (eval only, never trained on)

- `holdout_infiltration (ids2018)`: **701,039** rows, days ['Thursday-01-03-2018', 'Wednesday-28-02-2018']
- `holdout_ids2017_portscan_ddos (ids2017)`: **346,574** rows, days ['Friday-WorkingHours-Afternoon-DDos', 'Friday-WorkingHours-Afternoon-PortScan']

## Per-file processing (rows kept / duplicate rows dropped)

| source | file/day | rows kept | dup dropped | Inf/NaN dropped | holdout |
|---|---|---:|---:|---:|:--:|
| ids2018 | Wednesday-14-02-2018 | 816,188 | 228,558 | 3,824 |  |
| ids2018 | Thursday-15-02-2018 | 1,032,235 | 8,313 | 8,027 |  |
| ids2018 | Friday-16-02-2018 | 900,719 | 147,855 | 0 |  |
| ids2018 | Thuesday-20-02-2018 | 7,772,889 | 116,406 | 59,453 |  |
| ids2018 | Wednesday-21-02-2018 | 1,023,434 | 25,141 | 0 |  |
| ids2018 | Thursday-22-02-2018 | 1,034,647 | 8,309 | 5,610 |  |
| ids2018 | Friday-23-02-2018 | 1,035,029 | 7,838 | 5,708 |  |
| ids2018 | Wednesday-28-02-2018 | 402,112 | 204,790 | 6,169 | yes |
| ids2018 | Thursday-01-03-2018 | 298,927 | 29,254 | 2,919 | yes |
| ids2018 | Friday-02-03-2018 | 1,034,922 | 9,603 | 4,050 |  |
| ids2017 | Monday-WorkingHours | 515,525 | 13,941 | 452 |  |
| ids2017 | Tuesday-WorkingHours | 415,693 | 29,935 | 281 |  |
| ids2017 | Wednesday-workingHours | 606,090 | 85,295 | 1,318 |  |
| ids2017 | Thursday-WorkingHours-Morning-WebAttacks | 162,830 | 7,390 | 146 |  |
| ids2017 | Thursday-WorkingHours-Afternoon-Infilteration | 216,284 | 72,107 | 211 |  |
| ids2017 | Friday-WorkingHours-Morning | 181,640 | 9,262 | 131 |  |
| ids2017 | Friday-WorkingHours-Afternoon-DDos | 221,168 | 4,541 | 36 | yes |
| ids2017 | Friday-WorkingHours-Afternoon-PortScan | 125,406 | 160,654 | 407 | yes |

- Total duplicate rows dropped: **1,169,192**


## Checkpoint

- `train.isna().sum().sum() == 0`
- `holdout_infiltration (ids2018).isna().sum().sum() == 0`
- `holdout_ids2017_portscan_ddos (ids2017).isna().sum().sum() == 0`

## Class balance (merged train)

| mdp_state   |            rows |   percent |
|:------------|----------------:|----------:|
| Normal      |      1.435e+07  |     85.68 |
| Suspicious  | 168001          |      1    |
| Compromised |      2.2301e+06 |     13.32 |

### Per-source class balance (train)

| source_dataset   |      Normal |   Suspicious |      Compromised |       TOTAL |   Normal % |   Suspicious % |   Compromised % |
|:-----------------|------------:|-------------:|-----------------:|------------:|-----------:|---------------:|----------------:|
| ids2017          | 1.88952e+06 |        11358 | 197185           | 2.09806e+06 |      90.06 |           0.54 |            9.4  |
| ids2018          | 1.24605e+07 |       156643 |      2.03292e+06 | 1.46501e+07 |      85.05 |           1.07 |           13.88 |

### Per day x state (train)

| source_day                                    |   Normal |   Suspicious |   Compromised |
|:----------------------------------------------|---------:|-------------:|--------------:|
| Friday-02-03-2018                             |   753597 |            0 |        281325 |
| Friday-16-02-2018                             |   446635 |            0 |        454084 |
| Friday-23-02-2018                             |  1034473 |            0 |           556 |
| Friday-WorkingHours-Morning                   |   179787 |            0 |          1853 |
| Monday-WorkingHours                           |   515525 |            0 |             0 |
| Thuesday-20-02-2018                           |  7197162 |            0 |        575727 |
| Thursday-15-02-2018                           |   980801 |            0 |         51434 |
| Thursday-22-02-2018                           |  1034285 |            0 |           362 |
| Thursday-WorkingHours-Afternoon-Infilteration |   216248 |            0 |            36 |
| Thursday-WorkingHours-Morning-WebAttacks      |   160651 |            0 |          2179 |
| Tuesday-WorkingHours                          |   404335 |        11358 |             0 |
| Wednesday-14-02-2018                          |   659545 |       156643 |             0 |
| Wednesday-21-02-2018                          |   354005 |            0 |        669429 |
| Wednesday-workingHours                        |   412973 |            0 |        193117 |

## Attack label inventory (train)

|                                                             |     rows |
|:------------------------------------------------------------|---------:|
| ('ids2017', 'Compromised', 'DoS Hulk')                      |   171985 |
| ('ids2017', 'Compromised', 'DoS GoldenEye')                 |    10281 |
| ('ids2017', 'Compromised', 'DoS slowloris')                 |     5594 |
| ('ids2017', 'Compromised', 'DoS Slowhttptest')              |     5246 |
| ('ids2017', 'Compromised', 'Bot')                           |     1853 |
| ('ids2017', 'Compromised', 'Web Attack \x96 Brute Force')   |     1506 |
| ('ids2017', 'Compromised', 'Web Attack \x96 XSS')           |      652 |
| ('ids2017', 'Compromised', 'Infiltration')                  |       36 |
| ('ids2017', 'Compromised', 'Web Attack \x96 Sql Injection') |       21 |
| ('ids2017', 'Compromised', 'Heartbleed')                    |       11 |
| ('ids2017', 'Normal', 'BENIGN')                             |  1889519 |
| ('ids2017', 'Suspicious', 'FTP-Patator')                    |     6330 |
| ('ids2017', 'Suspicious', 'SSH-Patator')                    |     5028 |
| ('ids2018', 'Compromised', 'DDOS attack-HOIC')              |   667699 |
| ('ids2018', 'Compromised', 'DDoS attacks-LOIC-HTTP')        |   575727 |
| ('ids2018', 'Compromised', 'DoS attacks-Hulk')              |   434622 |
| ('ids2018', 'Compromised', 'Bot')                           |   281325 |
| ('ids2018', 'Compromised', 'DoS attacks-GoldenEye')         |    41437 |
| ('ids2018', 'Compromised', 'DoS attacks-SlowHTTPTest')      |    19462 |
| ('ids2018', 'Compromised', 'DoS attacks-Slowloris')         |     9997 |
| ('ids2018', 'Compromised', 'DDOS attack-LOIC-UDP')          |     1730 |
| ('ids2018', 'Compromised', 'Brute Force -Web')              |      601 |
| ('ids2018', 'Compromised', 'Brute Force -XSS')              |      230 |
| ('ids2018', 'Compromised', 'SQL Injection')                 |       87 |
| ('ids2018', 'Normal', 'Benign')                             | 12460503 |
| ('ids2018', 'Suspicious', 'SSH-Bruteforce')                 |   117297 |
| ('ids2018', 'Suspicious', 'FTP-BruteForce')                 |    39346 |

## 5-row sample trace (label -> MDP state)

| source_dataset   | source_day           | attack_label          | mdp_state   |   Dst Port Bucket |   Flow Pkts/s |   Active Conn Cnt | expected_state   |
|:-----------------|:---------------------|:----------------------|:------------|------------------:|--------------:|------------------:|:-----------------|
| ids2018          | Wednesday-14-02-2018 | Benign                | Normal      |         -0.594707 |     -0.141597 |         -0.7604   | Normal           |
| ids2018          | Wednesday-14-02-2018 | SSH-Bruteforce        | Suspicious  |         -0.594707 |      3.53623  |         -0.711573 | Suspicious       |
| ids2018          | Thursday-15-02-2018  | DoS attacks-GoldenEye | Compromised |         -0.594707 |     -0.141588 |         -0.655161 | Compromised      |
| ids2018          | Thursday-15-02-2018  | Benign                | Normal      |         -0.594707 |     -0.13621  |         -0.662272 | Normal           |
| ids2018          | Friday-23-02-2018    | Benign                | Normal      |         -0.594707 |     -0.141591 |         -0.635725 | Normal           |

All `mdp_state` values verified against `LABEL_TO_STATE` in the checkpoint tests.
