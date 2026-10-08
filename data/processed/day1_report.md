# Day 1 report - dataset preprocessing

- Files processed: **10**
- Files skipped (missing): none

- Train rows: **15,202,086**  |  Holdout rows: **935,083** (Infiltration days only)
- Train days: ['Friday-02-03-2018', 'Friday-16-02-2018', 'Friday-23-02-2018', 'Thuesday-20-02-2018', 'Thursday-15-02-2018', 'Thursday-22-02-2018', 'Wednesday-14-02-2018', 'Wednesday-21-02-2018']
- Holdout days: ['Thursday-01-03-2018', 'Wednesday-28-02-2018']
- Observation features (17): ['Flow Pkts/s', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Flow Byts/s', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Active Conn Cnt', 'Flow Duration', 'Pkt Len Mean', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Dst Port Bucket', 'Protocol']
- Hidden (reward/eval only): ['attack_label', 'mdp_state', 'TimestampEpoch', 'source_day']

## Checkpoint

- `train.isna().sum().sum() == 0`
- `holdout.isna().sum().sum() == 0`

## Class balance (train)

| mdp_state   |             rows |   percent |
|:------------|-----------------:|----------:|
| Normal      |      1.26158e+07 |     82.99 |
| Suspicious  | 380943           |      2.51 |
| Compromised |      2.20535e+06 |     14.51 |

### Per day x state (train)

| source_day           |   Normal |   Suspicious |   Compromised |
|:---------------------|---------:|-------------:|--------------:|
| Friday-02-03-2018    |   758334 |            0 |        286191 |
| Friday-16-02-2018    |   446772 |            0 |        601802 |
| Friday-23-02-2018    |  1042301 |            0 |           566 |
| Thuesday-20-02-2018  |  7313104 |            0 |        576191 |
| Thursday-15-02-2018  |   988050 |            0 |         52498 |
| Thursday-22-02-2018  |  1042594 |            0 |           362 |
| Wednesday-14-02-2018 |   663803 |       380943 |             0 |
| Wednesday-21-02-2018 |   360833 |            0 |        687742 |

## Attack label inventory

|                                             |   train_rows |   holdout_rows |
|:--------------------------------------------|-------------:|---------------:|
| ('Compromised', 'DDOS attack-HOIC')         |       686012 |              0 |
| ('Compromised', 'DDoS attacks-LOIC-HTTP')   |       576191 |              0 |
| ('Compromised', 'DoS attacks-Hulk')         |       461912 |              0 |
| ('Compromised', 'Bot')                      |       286191 |              0 |
| ('Compromised', 'DoS attacks-SlowHTTPTest') |       139890 |              0 |
| ('Compromised', 'DoS attacks-GoldenEye')    |        41508 |              0 |
| ('Compromised', 'DoS attacks-Slowloris')    |        10990 |              0 |
| ('Compromised', 'DDOS attack-LOIC-UDP')     |         1730 |              0 |
| ('Compromised', 'Brute Force -Web')         |          611 |              0 |
| ('Compromised', 'Brute Force -XSS')         |          230 |              0 |
| ('Compromised', 'SQL Injection')            |           87 |              0 |
| ('Normal', 'Benign')                        |     12615791 |         774444 |
| ('Suspicious', 'FTP-BruteForce')            |       193354 |              0 |
| ('Suspicious', 'SSH-Bruteforce')            |       187589 |              0 |
| ('Compromised', 'Infilteration')            |            0 |         160639 |

## 5-row sample trace (label -> MDP state)

| source_day           | attack_label          | mdp_state   |   Dst Port Bucket |   Flow Pkts/s |   Active Conn Cnt | expected_state   |
|:---------------------|:----------------------|:------------|------------------:|--------------:|------------------:|:-----------------|
| Wednesday-14-02-2018 | Benign                | Normal      |         -0.617332 |     -0.193004 |         -0.822425 | Normal           |
| Wednesday-14-02-2018 | SSH-Bruteforce        | Suspicious  |         -0.617332 |      2.36481  |         -0.768969 | Suspicious       |
| Thursday-15-02-2018  | DoS attacks-GoldenEye | Compromised |         -0.617332 |     -0.192997 |         -0.707209 | Compromised      |
| Thursday-15-02-2018  | Benign                | Normal      |          2.18094  |     -0.039535 |         -0.746393 | Normal           |
| Wednesday-21-02-2018 | Benign                | Normal      |          2.18094  |     -0.170454 |         -0.617164 | Normal           |

All `mdp_state` values verified against `LABEL_TO_STATE` in the checkpoint tests.
