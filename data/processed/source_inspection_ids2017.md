# Source inspection - ids2017

Directory: `data/raw/ids2017/MachineLearningCVE`
Files: **8**

- Total rows: **2,830,743**
- Exact duplicate rows: **256,479** (9.06%)

## Files

| file | rows | dup rows | labels (top 8) |
|---|---:|---:|---|
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | 225,745 | 2,633 | DDoS=128,027, BENIGN=97,718 |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | 286,467 | 72,353 | PortScan=158,930, BENIGN=127,537 |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | 191,033 | 6,888 | BENIGN=189,067, Bot=1,966 |
| Monday-WorkingHours.pcap_ISCX.csv | 529,918 | 26,935 | BENIGN=529,918 |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | 288,602 | 35,630 | BENIGN=288,566, Infiltration=36 |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | 170,366 | 6,066 | BENIGN=168,186, Web Attack � Brute Force=1,507, Web Attack � XSS=652, Web Attack � Sql Injection=21 |
| Tuesday-WorkingHours.pcap_ISCX.csv | 445,909 | 24,065 | BENIGN=432,074, FTP-Patator=7,938, SSH-Patator=5,897 |
| Wednesday-workingHours.pcap_ISCX.csv | 692,703 | 81,909 | BENIGN=440,031, DoS Hulk=231,073, DoS GoldenEye=10,293, DoS slowloris=5,796, DoS Slowhttptest=5,499, Heartbleed=11 |

## Header (first file, shown raw)

```
' Destination Port'
' Flow Duration'
' Total Fwd Packets'
' Total Backward Packets'
'Total Length of Fwd Packets'
' Total Length of Bwd Packets'
' Fwd Packet Length Max'
' Fwd Packet Length Min'
' Fwd Packet Length Mean'
' Fwd Packet Length Std'
'Bwd Packet Length Max'
' Bwd Packet Length Min'
' Bwd Packet Length Mean'
' Bwd Packet Length Std'
'Flow Bytes/s'
' Flow Packets/s'
' Flow IAT Mean'
' Flow IAT Std'
' Flow IAT Max'
' Flow IAT Min'
'Fwd IAT Total'
' Fwd IAT Mean'
' Fwd IAT Std'
' Fwd IAT Max'
' Fwd IAT Min'
'Bwd IAT Total'
' Bwd IAT Mean'
' Bwd IAT Std'
' Bwd IAT Max'
' Bwd IAT Min'
'Fwd PSH Flags'
' Bwd PSH Flags'
' Fwd URG Flags'
' Bwd URG Flags'
' Fwd Header Length'
' Bwd Header Length'
'Fwd Packets/s'
' Bwd Packets/s'
' Min Packet Length'
' Max Packet Length'
' Packet Length Mean'
' Packet Length Std'
' Packet Length Variance'
'FIN Flag Count'
' SYN Flag Count'
' RST Flag Count'
' PSH Flag Count'
' ACK Flag Count'
' URG Flag Count'
' CWE Flag Count'
' ECE Flag Count'
' Down/Up Ratio'
' Average Packet Size'
' Avg Fwd Segment Size'
' Avg Bwd Segment Size'
' Fwd Header Length.1'
'Fwd Avg Bytes/Bulk'
' Fwd Avg Packets/Bulk'
' Fwd Avg Bulk Rate'
' Bwd Avg Bytes/Bulk'
' Bwd Avg Packets/Bulk'
'Bwd Avg Bulk Rate'
'Subflow Fwd Packets'
' Subflow Fwd Bytes'
' Subflow Bwd Packets'
' Subflow Bwd Bytes'
'Init_Win_bytes_forward'
' Init_Win_bytes_backward'
' act_data_pkt_fwd'
' min_seg_size_forward'
'Active Mean'
' Active Std'
' Active Max'
' Active Min'
'Idle Mean'
' Idle Std'
' Idle Max'
' Idle Min'
' Label'
```

## Timestamp samples

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: []
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`: []
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`: []
- `Monday-WorkingHours.pcap_ISCX.csv`: []
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`: []
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`: []
- `Tuesday-WorkingHours.pcap_ISCX.csv`: []
- `Wednesday-workingHours.pcap_ISCX.csv`: []

## Non-finite numeric cells (sampled <=200k rows/file)

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: Flow Bytes/s=34,  Flow Packets/s=34
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`: Flow Bytes/s=327,  Flow Packets/s=327
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`: Flow Bytes/s=122,  Flow Packets/s=122
- `Monday-WorkingHours.pcap_ISCX.csv`: Flow Bytes/s=214,  Flow Packets/s=214
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`: Flow Bytes/s=196,  Flow Packets/s=196
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`: Flow Bytes/s=135,  Flow Packets/s=135
- `Tuesday-WorkingHours.pcap_ISCX.csv`: Flow Bytes/s=163,  Flow Packets/s=163
- `Wednesday-workingHours.pcap_ISCX.csv`: Flow Bytes/s=735,  Flow Packets/s=735

## Combined label inventory

| label | rows |
|---|---:|
| 'BENIGN' | 2,273,097 |
| 'DoS Hulk' | 231,073 |
| 'PortScan' | 158,930 |
| 'DDoS' | 128,027 |
| 'DoS GoldenEye' | 10,293 |
| 'FTP-Patator' | 7,938 |
| 'SSH-Patator' | 5,897 |
| 'DoS slowloris' | 5,796 |
| 'DoS Slowhttptest' | 5,499 |
| 'Bot' | 1,966 |
| 'Web Attack � Brute Force' | 1,507 |
| 'Web Attack � XSS' | 652 |
| 'Infiltration' | 36 |
| 'Web Attack � Sql Injection' | 21 |
| 'Heartbleed' | 11 |
