| méthode | h=2 (m=8) | h=3 (m=12) | h=4 (m=16) |
|---|---|---|---|
| nelder-mead | 53 % (560) | 53 % (745) | 20 % (948) |
| powell | 3 % (1325) | 0 % (-) | 0 % (-) |
| bfgs-fd | 20 % (838) | 43 % (1236) | 7 % (1370) |
| gauss-newton-fd (trf) | 13 % (1059) | 20 % (1161) | 3 % (889) |
| cma-es | 70 % (801) | 77 % (1028) | 10 % (1333) |
| adam-fd | 0 % (-) | 0 % (-) | 0 % (-) |
| sonde-newton-8b | 23 % (650) | 27 % (684) | 7 % (991) |
| sonde-newton-12b | 43 % (784) | 37 % (781) | 20 % (1098) |
| sonde-broyden-8b | 0 % (-) | 0 % (-) | 0 % (-) |
| sonde-broyden-12b | 0 % (-) | 0 % (-) | 0 % (-) |

Succès % (évaluations médianes parmi les succès). Réglages choisis :
- nelder-mead : h=2: None, h=3: None, h=4: None
- powell : h=2: None, h=3: None, h=4: None
- bfgs-fd : h=2: None, h=3: None, h=4: None
- gauss-newton-fd (trf) : h=2: None, h=3: None, h=4: None
- cma-es : h=2: 0.3, h=3: 0.3, h=4: 0.3
- adam-fd : h=2: 0.01, h=3: 0.01, h=4: 0.01
- sonde-newton-8b : h=2: [0.7, 0.3], h=3: [0.7, 0.1], h=4: [0.3, 0.1]
- sonde-newton-12b : h=2: [0.7, 0.1], h=3: [0.7, 0.1], h=4: [1.0, 0.1]
- sonde-broyden-8b : h=2: [0.7, 0.3], h=3: [0.7, 0.3], h=4: [0.7, 0.3]
- sonde-broyden-12b : h=2: [0.7, 0.3], h=3: [0.7, 0.3], h=4: [0.7, 0.3]
