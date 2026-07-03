# WRS-v2 Ranking Protocol

This project uses WRS-v2 as the main watermarking leaderboard protocol and
keeps a WAVES-style attack leaderboard for attack-specific diagnosis.

## Main Leaderboard

WRS-v2 ranks watermarking algorithms by robustness under a physical-aware
attack taxonomy. Each completed experiment cell contributes:

- `TPR@FPR=0.001` as the detection performance value.
- `NQD` as normalized quality degradation.
- normalized attack strength in `[0, 1]`.
- an attack family category.

For each algorithm and each attack family, WRS-v2 computes the area under the
performance curve over normalized attack strength. Cells are included only when
they are practical for ranking:

- the attack family is part of the WRS-v2 taxonomy;
- `TPR@FPR` is available;
- `NQD` is available and below the practical-quality threshold.

The main `wrs` score is the mean of covered family AUC values scaled to 0-100.
Runs remain `provisional` until every required family is covered and the sample
floor is met.

## Attack Taxonomy

The WRS-v2 attack families are:

- `distortion-single`
- `distortion-combination`
- `content-preserving-workflow`
- `consumer-enhancement-workflow`
- `regeneration`
- `physical-screen`
- `physical-print`
- `physical-combined`
- `adversarial`

The physical families are intentionally separate. This keeps screen-shoot,
print-camera, and two-hop combined physical failure modes visible instead of
hiding them inside one averaged physical bucket.

## Method Profiles

The leaderboard includes secondary fields for interpretation:

- `physicalScore`: mean of `physical-screen`, `physical-print`, and
  `physical-combined` scores when present.
- `worstCategory`: the weakest covered attack family.
- `profileTags`: short labels such as `physical-robust`,
  `screen-specialist`, `print-specialist`, `combined-fragile`,
  `quality-first`, `fast-lightweight`, and `geometry-fragile`.

These fields explain why an algorithm ranks where it does. They should not
replace the primary WRS-v2 rank.

## Attack Leaderboard

The attack leaderboard keeps the WAVES-style `Q@P + AvgP/AvgQ` view:

- `Q@0.95P`: quality degradation at the point where performance first falls to
  0.95.
- `Q@0.7P`: quality degradation at the point where performance first falls to
  0.70.
- `Avg P`: mean performance over available attack points.
- `Avg Q`: mean NQD over available attack points.
- `AUC`: performance AUC over normalized attack strength.

This table answers a different question from the main leaderboard: which attack
families and strengths expose the failure surface of each algorithm.

## UI Behavior

The Results page can select any run discovered in the database or under the
artifact `runs/` directory. The overview tab shows WRS-v2 bars, a radar profile,
coverage, and method tags. The attack tab shows the WAVES-style attack
leaderboard. The quality tab shows interactive curve points. Clicking bars,
radar points, table rows, or curve points updates the insight strip above the
charts.

## Offline Exports

Use these scripts after experiment artifacts are available:

```powershell
python 算法\scripts\build_watermark_rankings.py --runs-root runs --out runs\rankings
python 算法\scripts\plot_watermark_rankings.py --summary runs\rankings\ranking_summary.json --out runs\rankings\figures
```

The export script writes:

- `leaderboard.csv`
- `attack_leaderboard.csv`
- `method_profiles.csv`
- `ranking_summary.json`

The plotting script writes PNG figures for quick report review.
