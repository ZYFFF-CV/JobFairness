# Stage2 Probe Protocol Audit

Status: row-disjoint and raw-user-disjoint samples frozen.

The Stage2 probe protocol retains the original sequential FairJob outer split.
It does not reshuffle CTR train/test rows. Probe train/validation/test roles are
separate evaluation subsets used only after representation export.

## Row-disjoint Sample

The inherited Stage1.1 sample contains 50,000 frozen rows from each outer data
split:

```text
/root/autodl-tmp/workdirs/JobFairness/stage1_1/probes/probe_rows.npz
```

SHA-256:
`d8ee59167b2ae0a03fbd16e6bdfa478fbf044c0070f4150239a9061b22de2140`.

## Raw-user-disjoint Sample

Stage2 assigns raw `user_id` values by a deterministic BLAKE2b hash with seed
2019 and a 60/20/20 probe train/validation/test ratio. FuxiCTR tokenizer IDs
are never used for identity separation.

The frozen output is:

```text
/root/autodl-tmp/workdirs/JobFairness/stage2/manifests/user_disjoint_probe_rows.npz
```

| Probe split | Rows | Raw users | Group 0 rows | Group 1 rows |
| --- | ---: | ---: | ---: | ---: |
| train | 29,939 | 7,361 | 15,245 | 14,694 |
| validation | 9,881 | 2,574 | 4,738 | 5,143 |
| test | 10,217 | 2,455 | 5,235 | 4,982 |

Pairwise raw-user overlap is exactly zero for train/validation, train/test, and
validation/test. The source CSV and sample hashes are retained in the NPZ
manifest generated at commit
`f80d8c8fb790ddecb8335b01aee0dd69b5a56b6f`.

Probe capacity and AUC orientation are selected on validation only. Test is
read once for each selected linear, matched-MLP, and bounded independent
ExtraTrees family. These probes support a bounded behavioral decodability
claim, not proof of information erasure.
