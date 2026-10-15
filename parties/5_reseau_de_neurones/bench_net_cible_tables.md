
### Faire aussi bien que le professeur à 10 % près — succès % (évaluations médianes)

| méthode | 8 paramètres | 12 paramètres | 16 paramètres | 24 paramètres | moyenne |
|---|---|---|---|---|---|
| sonde-cible-12b | 70 (176) | 100 (411) | 55 (1913) | 100 (1438) | 81 |
| sonde-cible-8b | 70 (262) | 95 (1131) | 20 (2780) | 100 (3527) | 71 |
| même algorithme, gradient exact | 70 (177) | 100 (379) | 70 (2303) | 100 (1201) | 85 |
| bfgs (différences finies) | 75 (136) | 100 (287) | 60 (1658) | 100 (976) | 84 |
| bfgs (rétropropagation) | 75 (31) | 100 (45) | 60 (195) | 100 (79) | 84 |
| nelder-mead | 45 (1166) | 75 (1661) | 5 (4126) | 45 (5201) | 42 |
| cma-es | 80 (622) | 90 (828) | 35 (2443) | 100 (2724) | 76 |
| adam (différences finies, ln ℓ) | 85 (1387) | 100 (1015) | 50 (1990) | 100 (4051) | 84 |
| adam (rétropropagation) | 85 (477) | 100 (781) | 50 (2543) | 100 (2764) | 84 |
| sonde-teinte-12b, courbure diagonale | 80 (186) | 95 (285) | 30 (1099) | 95 (1031) | 75 |
| sonde-teinte-12b, échelle de courbure | 85 (134) | 95 (241) | 55 (1541) | 95 (631) | 82 |
| sonde-teinte-12b, courbure non utilisée | 70 (193) | 100 (399) | 45 (1793) | 100 (1276) | 79 |
| sonde-teinte-12b, courbure diagonale, λ sur la médiane (premier essai) | 40 (171) | 60 (549) | 15 (2040) | 85 (1513) | 50 |
| sonde-teinte-12b, forme polaire, échelle de courbure | 75 (142) | 100 (175) | 75 (1104) | 100 (704) | 88 |
| sonde-teinte-12b, forme polaire, courbure diagonale | 90 (170) | 95 (264) | 30 (1200) | 100 (881) | 79 |
| sonde-teinte-8b, forme polaire, échelle de courbure | 45 (220) | 70 (228) | 0 (–) | 45 (1261) | 40 |

### Faire aussi bien que le professeur à 1 % près — succès % (évaluations médianes)

| méthode | 8 paramètres | 12 paramètres | 16 paramètres | 24 paramètres | moyenne |
|---|---|---|---|---|---|
| sonde-cible-12b | 70 (222) | 100 (502) | 50 (2058) | 85 (1982) | 76 |
| sonde-cible-8b | 70 (307) | 90 (1212) | 15 (3307) | 70 (3486) | 61 |
| même algorithme, gradient exact | 70 (235) | 100 (454) | 70 (2382) | 100 (2164) | 85 |
| bfgs (différences finies) | 75 (163) | 100 (352) | 60 (1752) | 100 (1401) | 84 |
| bfgs (rétropropagation) | 75 (37) | 100 (55) | 60 (207) | 100 (113) | 84 |
| nelder-mead | 45 (1242) | 70 (1570) | 0 (–) | 25 (5946) | 35 |
| cma-es | 80 (683) | 90 (985) | 30 (2344) | 90 (3268) | 72 |
| adam (différences finies, ln ℓ) | 80 (1423) | 100 (1126) | 50 (2100) | 90 (4764) | 80 |
| adam (rétropropagation) | 80 (593) | 95 (1213) | 45 (2763) | 85 (3651) | 76 |
| sonde-teinte-12b, courbure diagonale | 80 (216) | 95 (394) | 25 (1140) | 95 (1709) | 74 |
| sonde-teinte-12b, échelle de courbure | 85 (157) | 95 (275) | 55 (1631) | 95 (1036) | 82 |
| sonde-teinte-12b, courbure non utilisée | 70 (236) | 100 (525) | 45 (2585) | 100 (2226) | 79 |
| sonde-teinte-12b, courbure diagonale, λ sur la médiane (premier essai) | 35 (181) | 55 (692) | 10 (1446) | 80 (2006) | 45 |
| sonde-teinte-12b, forme polaire, échelle de courbure | 75 (155) | 100 (250) | 65 (1072) | 100 (954) | 85 |
| sonde-teinte-12b, forme polaire, courbure diagonale | 90 (178) | 95 (327) | 30 (1302) | 95 (1157) | 77 |
| sonde-teinte-8b, forme polaire, échelle de courbure | 45 (229) | 70 (290) | 0 (–) | 25 (1473) | 35 |

### Faire aussi bien que le professeur à 0.1 % près — succès % (évaluations médianes)

| méthode | 8 paramètres | 12 paramètres | 16 paramètres | 24 paramètres | moyenne |
|---|---|---|---|---|---|
| sonde-cible-12b | 70 (222) | 100 (502) | 50 (2068) | 85 (2061) | 76 |
| sonde-cible-8b | 70 (307) | 90 (1274) | 15 (3324) | 65 (3568) | 60 |
| même algorithme, gradient exact | 70 (235) | 100 (496) | 70 (2400) | 100 (2352) | 85 |
| bfgs (différences finies) | 75 (163) | 100 (365) | 60 (1760) | 100 (1514) | 84 |
| bfgs (rétropropagation) | 75 (37) | 100 (57) | 60 (210) | 100 (122) | 84 |
| nelder-mead | 45 (1265) | 70 (1579) | 0 (–) | 20 (5887) | 34 |
| cma-es | 80 (690) | 90 (1006) | 30 (2366) | 90 (3482) | 72 |
| adam (différences finies, ln ℓ) | 80 (1428) | 100 (1138) | 50 (2168) | 85 (4751) | 79 |
| adam (rétropropagation) | 80 (613) | 95 (1267) | 45 (2803) | 85 (3811) | 76 |
| sonde-teinte-12b, courbure diagonale | 80 (218) | 95 (394) | 25 (1259) | 95 (1709) | 74 |
| sonde-teinte-12b, échelle de courbure | 85 (175) | 95 (275) | 55 (1648) | 95 (1078) | 82 |
| sonde-teinte-12b, courbure non utilisée | 70 (240) | 100 (525) | 45 (2621) | 100 (2376) | 79 |
| sonde-teinte-12b, courbure diagonale, λ sur la médiane (premier essai) | 35 (181) | 55 (726) | 10 (1460) | 80 (2072) | 45 |
| sonde-teinte-12b, forme polaire, échelle de courbure | 75 (169) | 100 (276) | 65 (1089) | 100 (1029) | 85 |
| sonde-teinte-12b, forme polaire, courbure diagonale | 90 (183) | 95 (353) | 30 (1320) | 95 (1207) | 77 |
| sonde-teinte-8b, forme polaire, échelle de courbure | 45 (229) | 65 (279) | 0 (–) | 25 (1473) | 34 |
