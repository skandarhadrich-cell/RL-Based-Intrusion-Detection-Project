# Source inspection - ids2017_trafficlabelling

Directory: `data/raw/ids2017/TrafficLabelling`
Files: **8**

- Total rows: **3,119,345**
- Exact duplicate rows: **288,803** (9.26%)

## Files

| file | rows | dup rows | labels (top 8) |
|---|---:|---:|---|
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | 225,745 | 2 | DDoS=128,027, BENIGN=97,718 |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | 286,467 | 1 | PortScan=158,930, BENIGN=127,537 |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | 191,033 | 2 | BENIGN=189,067, Bot=1,966 |
| Monday-WorkingHours.pcap_ISCX.csv | 529,918 | 34 | BENIGN=529,918 |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | 288,602 | 142 | BENIGN=288,566, Infiltration=36 |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | 458,968 | 288,601 | nan=288,602, BENIGN=168,186, Web Attack  Brute Force=1,507, Web Attack  XSS=652, Web Attack  Sql Injection=21 |
| Tuesday-WorkingHours.pcap_ISCX.csv | 445,909 | 4 | BENIGN=432,074, FTP-Patator=7,938, SSH-Patator=5,897 |
| Wednesday-workingHours.pcap_ISCX.csv | 692,703 | 17 | BENIGN=440,031, DoS Hulk=231,073, DoS GoldenEye=10,293, DoS slowloris=5,796, DoS Slowhttptest=5,499, Heartbleed=11 |

## Header (first file, shown raw)

```
'Flow ID'
' Source IP'
' Source Port'
' Destination IP'
' Destination Port'
' Protocol'
' Timestamp'
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

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: ['7/7/2017 3:30']
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`: ['7/7/2017 1:00', '7/7/2017 3:09']
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`: ['7/7/2017 8:59', '7/7/2017 9:00']
- `Monday-WorkingHours.pcap_ISCX.csv`: ['03/07/2017 08:55:58', '03/07/2017 08:56:22', '03/07/2017 01:30:56']
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`: ['6/7/2017 1:00', '6/7/2017 4:19']
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`: ['6/7/2017 8:59']
- `Tuesday-WorkingHours.pcap_ISCX.csv`: ['4/7/2017 8:54', '4/7/2017 4:26']
- `Wednesday-workingHours.pcap_ISCX.csv`: ['5/7/2017 8:42', '5/7/2017 10:55', '5/7/2017 4:23']

## Non-finite numeric cells (sampled <=200k rows/file)

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: Flow Bytes/s=34,  Flow Packets/s=34
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`: Flow Bytes/s=327,  Flow Packets/s=327
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`: Flow Bytes/s=122,  Flow Packets/s=122
- `Monday-WorkingHours.pcap_ISCX.csv`: Flow Bytes/s=214,  Flow Packets/s=214
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`: Flow Bytes/s=196,  Flow Packets/s=196
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`: Flow Bytes/s=79,769,  Flow Packets/s=79,769,  Source Port=79,634,  Destination Port=79,634,  Protocol=79,634,  Flow Duration=79,634,  Total Fwd Packets=79,634,  Total Backward Packets=79,634, Total Length of Fwd Packets=79,634,  Total Length of Bwd Packets=79,634,  Fwd Packet Length Max=79,634,  Fwd Packet Length Min=79,634,  Fwd Packet Length Mean=79,634,  Fwd Packet Length Std=79,634, Bwd Packet Length Max=79,634,  Bwd Packet Length Min=79,634,  Bwd Packet Length Mean=79,634,  Bwd Packet Length Std=79,634,  Flow IAT Mean=79,634,  Flow IAT Std=79,634,  Flow IAT Max=79,634,  Flow IAT Min=79,634, Fwd IAT Total=79,634,  Fwd IAT Mean=79,634,  Fwd IAT Std=79,634,  Fwd IAT Max=79,634,  Fwd IAT Min=79,634, Bwd IAT Total=79,634,  Bwd IAT Mean=79,634,  Bwd IAT Std=79,634,  Bwd IAT Max=79,634,  Bwd IAT Min=79,634, Fwd PSH Flags=79,634,  Bwd PSH Flags=79,634,  Fwd URG Flags=79,634,  Bwd URG Flags=79,634,  Fwd Header Length=79,634,  Bwd Header Length=79,634, Fwd Packets/s=79,634,  Bwd Packets/s=79,634,  Min Packet Length=79,634,  Max Packet Length=79,634,  Packet Length Mean=79,634,  Packet Length Std=79,634,  Packet Length Variance=79,634, FIN Flag Count=79,634,  SYN Flag Count=79,634,  RST Flag Count=79,634,  PSH Flag Count=79,634,  ACK Flag Count=79,634,  URG Flag Count=79,634,  CWE Flag Count=79,634,  ECE Flag Count=79,634,  Down/Up Ratio=79,634,  Average Packet Size=79,634,  Avg Fwd Segment Size=79,634,  Avg Bwd Segment Size=79,634,  Fwd Header Length.1=79,634, Fwd Avg Bytes/Bulk=79,634,  Fwd Avg Packets/Bulk=79,634,  Fwd Avg Bulk Rate=79,634,  Bwd Avg Bytes/Bulk=79,634,  Bwd Avg Packets/Bulk=79,634, Bwd Avg Bulk Rate=79,634, Subflow Fwd Packets=79,634,  Subflow Fwd Bytes=79,634,  Subflow Bwd Packets=79,634,  Subflow Bwd Bytes=79,634, Init_Win_bytes_forward=79,634,  Init_Win_bytes_backward=79,634,  act_data_pkt_fwd=79,634,  min_seg_size_forward=79,634, Active Mean=79,634,  Active Std=79,634,  Active Max=79,634,  Active Min=79,634, Idle Mean=79,634,  Idle Std=79,634,  Idle Max=79,634,  Idle Min=79,634
- `Tuesday-WorkingHours.pcap_ISCX.csv`: Flow Bytes/s=163,  Flow Packets/s=163
- `Wednesday-workingHours.pcap_ISCX.csv`: Flow Bytes/s=735,  Flow Packets/s=735

## Combined label inventory

| label | rows |
|---|---:|
| 'BENIGN' | 2,273,097 |
| nan | 288,602 |
| 'DoS Hulk' | 231,073 |
| 'PortScan' | 158,930 |
| 'DDoS' | 128,027 |
| 'DoS GoldenEye' | 10,293 |
| 'FTP-Patator' | 7,938 |
| 'SSH-Patator' | 5,897 |
| 'DoS slowloris' | 5,796 |
| 'DoS Slowhttptest' | 5,499 |
| 'Bot' | 1,966 |
| 'Web Attack \x96 Brute Force' | 1,507 |
| 'Web Attack \x96 XSS' | 652 |
| 'Infiltration' | 36 |
| 'Web Attack \x96 Sql Injection' | 21 |
| 'Heartbleed' | 11 |
