# v2 strut classification — results

18468 design struts.  Verdict counts (all / interior-context only):

| verdict | all | % | interior |
|---|---:|---:|---:|
| present | 16646 | 90.13% | 12112 |
| missing | 383 | 2.07% | 74 |
| disconnected | 569 | 3.08% | 31 |
| thin | 548 | 2.97% | 419 |
| bent | 322 | 1.74% | 116 |

**defective: 1822 (9.87%)**

## missing, split by cause
- dropped strut (both nodes printed): **354** — graph-confirmed (hops>=2): 349/349 with both nodes in the graph
- node_lost (one endpoint node never printed): **29**
- void (both endpoint nodes never printed): **0**

disconnected with a lost node: 3

## thin, split by cause
- hairline (whole strut below cutoff): 36
- necked (locally below cutoff, median normal): 512

## audits
- phantom edges vetoed (stub tip merged into a node): 462
- corridor-blind but skeleton path locally fused to both cores (trusted edge): 204
- fragmented edges where the corridor verdict stood: 2
- present bow um (med/p90): 69/117 — bent bow um (med): 284
- present dia um (med): 329 — thin dia um (med): 232

Every strut record carries its own measurements (profile string, gap, stub,
bow, diameter, node states, graph hops) — validation panels are drawn from
these records, so a panel cannot disagree with its verdict.