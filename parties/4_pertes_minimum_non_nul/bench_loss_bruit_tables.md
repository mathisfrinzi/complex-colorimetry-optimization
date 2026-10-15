
### Tolérance 0.01 — succès % (évaluations médianes), moyenne sur 4 familles et n = 2, 5, 10

| méthode | σ = 0 | σ = 0.0001 | σ = 0.001 | σ = 0.01 |
|---|---|---|---|---|
| sonde-cible-avant-12b | 97 (103) | 42 (34) | 9 (19) | 2 (22) |
| sonde-cible-centrée-12b | 96 (132) | 57 (82) | 29 (43) | 3 (61) |
| bfgs | 94 (100) | 56 (55) | 13 (29) | 0 (nan) |
| nelder-mead | 89 (282) | 80 (266) | 59 (126) | 19 (44) |
| cma-es | 90 (375) | 89 (369) | 87 (350) | 79 (322) |
| adam-ln-fd | 93 (667) | 73 (715) | 18 (360) | 2 (288) |
| sonde-cible-plancher-12b | 97 (134) | 88 (137) | 54 (142) | 1 (21) |

### Tolérance 0.001 — succès % (évaluations médianes), moyenne sur 4 familles et n = 2, 5, 10

| méthode | σ = 0 | σ = 0.0001 | σ = 0.001 | σ = 0.01 |
|---|---|---|---|---|
| sonde-cible-avant-12b | 97 (115) | 20 (42) | 2 (28) | 0 (nan) |
| sonde-cible-centrée-12b | 96 (161) | 32 (63) | 9 (44) | 0 (28) |
| bfgs | 94 (127) | 24 (43) | 6 (30) | 0 (nan) |
| nelder-mead | 89 (312) | 78 (281) | 52 (106) | 14 (51) |
| cma-es | 88 (450) | 88 (450) | 86 (425) | 49 (210) |
| adam-ln-fd | 85 (859) | 43 (576) | 10 (394) | 0 (127) |
| sonde-cible-plancher-12b | 97 (156) | 70 (186) | 38 (202) | 0 (28) |
