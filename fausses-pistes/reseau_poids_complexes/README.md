# Fausse piste : poids complexes holomorphes lus en couleur

**Idée.** Avec des poids complexes et une activation holomorphe (`tanh` complexe), chaque résidu `rᵢ(θ)` est holomorphe en θ. Comme pour la sonde unique de la partie 1, **une seule sonde par paramètre complexe** devrait suffire à lire toute la colonne du jacobien, au lieu de deux si l'on traite parties réelle et imaginaire séparément. La perte `Σ|rᵢ|²` est réelle, donc jamais holomorphe : pour que l'holomorphie serve, on lit **les résidus un par un** (une couleur par donnée, comme en partie 3) et on fait un pas de Levenberg-Marquardt.

**Résultat.** Le principe est juste (avec un jacobien exact, l'holomorphie divise le coût par 1,7 à 1,9), mais le jacobien **lu en couleur** est trop imprécis : 0 à 10 % de succès.

## Montage

`bench_net_holo.py`, `src/sonde_residus.py` : réseau 2 → h `tanh` complexes → 1 sortie complexe, 4h paramètres complexes (h = 2, 3, 4) ; 100 points, cibles complexes = professeur + bruit ; critère : faire aussi bien que le professeur à 1 % près ; 20 départs ; budget 300 × (paramètres réels).

## Résultats : succès à 1 % près (évaluations médianes)

| méthode | 8 param. complexes | 12 | 16 |
|---|---|---|---|
| **Levenberg-Marquardt, jacobien exact, holomorphie exploitée** | **50 % (206)** | **10 % (212)** | **40 % (737)** |
| Levenberg-Marquardt, jacobien exact, sans holomorphie | 50 % (378) | 10 % (398) | 30 % (1234) |
| scipy `least_squares` (lm), différences finies | 60 % (365) | 25 % (686) | 40 % (1501) |
| BFGS, différences finies | 65 % (1497) | 20 % (2826) | 30 % (7674) |
| jacobien lu en couleur (12 bits), holomorphie exploitée | 0 % | 5 % (440) | 10 % (710) |
| jacobien lu en couleur (12 bits), sans holomorphie | 0 % | 10 % (834) | 5 % (1182) |
| jacobien lu en couleur (8 bits) | 0 % | 0 % | 0 % |

## Pourquoi la couleur échoue ici

1. **Le principe est confirmé** : avec un jacobien exact, exploiter l'holomorphie divise le coût par 1,7 à 1,9 (206 contre 378, 212 contre 398, 737 contre 1 234) au même succès, et rend la méthode moins chère que scipy et BFGS.
2. **La couleur lit des variations relatives** de chaque résidu : l'erreur sur une dérivée vaut environ `|rᵢ|·ε/η`, énorme pour les grands résidus (le `tanh` complexe a des pôles, certains résidus explosent). Mesuré : 2 à 60 % d'erreur sur le jacobien. Une fois, le gradient lu pointait même à l'envers (cosinus −0,84) : les résidus dont la couleur avait « sauté » étaient écartés, et c'étaient eux qui portaient le gradient.
3. En partie 3, le même capteur marchait parce que le pas de Newton `W h = −1` ne fait intervenir que des variations relatives. Ici, les moindres carrés mélangent des résidus de tailles très différentes.
4. Le problème est difficile en soi (pôles, minima locaux) : même le meilleur Levenberg-Marquardt ne réussit que 10 à 60 % des départs.

Une piste pour la sauver est notée dans [`IDEES_A_EXPLORER.md`](../../IDEES_A_EXPLORER.md) : lire `ln rᵢ` et résoudre en variables relatives.
