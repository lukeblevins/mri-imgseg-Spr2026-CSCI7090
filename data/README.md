# External Dataset Root

This directory is intentionally kept out of version control except for this file.

Place benchmark datasets here only for local development. The expected layout is:

```text
data/
  chbmit/chb01/chb01_01.edf
  siena/pn00/PN00-1.edf
```

You can also keep the datasets elsewhere and point the workflows to them with `EEG_DATA_ROOT`.
