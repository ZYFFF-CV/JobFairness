# Stage2 DCNv2 Representation Graph Audit

Status: implementation-frozen; server dry-run evidence pending.

## Scope

Stage2 uses the native FuxiCTR `DCNv2` implementation with
`model_structure=parallel`, three `CrossNetV2` layers, and a two-block parallel
DNN. No code under `model_zoo/` or `fuxictr/` is modified.

The executable graph definition is
`configs/fairjob/stage2_representation_graph.yaml`. Its stable hash and actual
forward tensor shapes are written into every Stage2 dry-run/run manifest.

## Actual Computation Graph

```text
model inputs
  -> feature embedding
     -> CrossNetV2 increment 0 -> cross layer 0
     -> CrossNetV2 increment 1 -> cross layer 1
     -> CrossNetV2 increment 2 -> cross layer 2
     -> cross branch gate

  -> feature embedding
     -> parallel DNN linear 0
     -> parallel DNN linear 1
     -> DNN branch gate

cross branch + DNN branch
  -> concatenated fusion representation
  -> native DCNv2 output linear
  -> click probability
```

The Stage2 gates act on whole named increments, DNN blocks, or branches. They
never assign semantic meaning to hidden coordinates. `suppressed_component`
and `residual_component` from Stage1.1 are retained only as legacy controls and
are not treated as native DCNv2 paths.

## Invariants

- `embedding_flat` is a sentinel, not an adversarial target.
- Cross increments retain the native equation
  `x0 * linear_i(x_i)` before their scalar structural gate.
- `fusion_pre_logit` and `dcnv2_final` are exact aliases.
- Joint paths concatenate row-aligned members in a frozen order.
- Raw `protected_attribute_meta` is excluded by `BaseModel.get_inputs()`.
- Proxy metadata is read only while training an adversarial method.
- Evaluation and inference run without a protected proxy input.
- Formal methods initialize from and retain a frozen same-seed baseline
  checkpoint for anti-collapse preservation losses.

## Automatic Audit

`run_fuxictr_model.py --dry_run` validates one real batch for:

- all frozen graph nodes being present;
- equal row counts;
- finite tensor values;
- declared alias equality;
- actual tensor dimensions;
- prediction equality between `forward()` and representation export;
- graph and gate definition hashes in the run manifest.

The final section of this document will be updated with the server manifest
after the seed-2019 bounded dry-run.

## Claim Boundary

The graph supports behavioral protected-proxy decodability experiments. It
does not establish information erasure, causal pathways, verified demographic
attributes, or fairness under an unavailable position-propensity model.
