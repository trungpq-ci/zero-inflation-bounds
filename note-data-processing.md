**TL;DR**:
1. Fix data processing bug
2. Run standard Zero-inflation method for CLABSI on selected data

# Data

Use only PatientData in this experiment

1. `PatientData_20230315`
2. Guideline files `CLABSI-grouping-LineType.csv` and `CLABSI-grouping-TherapyType.csv` for string processing.

## Data Processing

Fix the bug in string processing. Previously

```
df[col] = df[col].str.lower().str.strip().str.replace(",", "/").str.replace(";", "/").str.replace("\n", "\\")
```

So a raw string `tpn\nhydration` turns into raw string `tpn\\hydration`, displayed as "tpn\hydration". However, the guideline file has "tpn\\hydration", which is `tpn\\\\hydration` in raw string. And hence this fails to match. Fix this as

```
df[col] = df[col].str.lower().str.strip().str.replace(",", "/").str.replace(";", "/").str.replace("\n", "//")
```