# Fausse piste : deux pertes redondantes dans un nombre complexe

**Idée.** Coder deux pertes dans `F = L_A + i·L_B`, avec `L_A` et `L_B` les erreurs sur deux moitiés des données, et chercher un zéro de `F` avec la sonde. On espérait « deux informations à réduire pour le prix d'une lecture ».

**Résultat : négatif.** Les deux pertes disent presque la même chose, et le problème devient très sous-déterminé. C'est ce qui a conduit, en partie 5, à garder la perte usuelle et à donner à la partie imaginaire une information qui la complète vraiment (courbure, perte de validation).

## Montage (`bench_net.py`)

- **Réseau** (élève et professeur, même architecture) : 2 entrées → `h` neurones `tanh` → 1 sortie linéaire sans biais ; `m = 4h` paramètres réels (`h = 2, 3, 4` → 8, 12, 16). Professeur tiré selon une loi normale (graine 0), élève tiré indépendamment.
- **Données** : 60 points gaussiens en 2D, cible = sortie exacte du professeur (pas de bruit) : un zéro exact existe.
- **Pertes** : `L_A` = erreur quadratique moyenne sur les 30 premiers points, `L_B` sur les 30 derniers ; la teinte de `F` code le rapport `L_B/L_A`, sa luminosité `ln|F|`.
- **Sonde** : paramètres réels, donc pas d'holomorphie : `m + 1` lectures pour estimer `W = ∂ ln F/∂θ`, puis pas de norme minimale vérifiant `Re(W)·h = −1` et `Im(W)·h = 0`. Variante Broyden : 1 lecture par pas.
- **Concurrents** (sur `L_A² + L_B²`, double précision) : Nelder-Mead, Powell, BFGS et Gauss-Newton par différences finies, CMA-ES, Adam par différences finies.
- **Protocole** : succès `max(L_A, L_B) < 10⁻³`, budget 1 500, 30 départs, une graine.

## Résultats : succès % (évaluations médianes)

| méthode | m = 8 | m = 12 | m = 16 |
|---|---|---|---|
| Nelder-Mead | 53 (560) | 53 (745) | 20 (948) |
| Powell | 3 | 0 | 0 |
| BFGS-FD | 20 | 43 | 7 |
| Gauss-Newton FD | 13 | 20 | 3 |
| CMA-ES | 70 (801) | 77 (1028) | 10 (1333) |
| Adam-FD | 0 | 0 | 0 |
| sonde-Newton 8 bits | 23 (650) | 27 (684) | 7 (991) |
| sonde-Newton 12 bits | 43 (784) | 37 (781) | 20 (1098) |
| sonde-Broyden 8 / 12 bits | 0 | 0 | 0 |

## Pourquoi ça ne marche pas

1. **Les deux pertes sont redondantes.** Un réseau qui colle à la moitié A colle presque toujours à la moitié B : la deuxième information n'apporte presque rien.
2. **Le système est très sous-déterminé** : un nombre complexe, soit 2 équations réelles, pour 8 à 16 inconnues. Les solutions forment une variété et `W` est presque singulière près du zéro ; la sonde cale vers une perte de 5·10⁻³.
3. **Les pertes s'annulent quadratiquement**, donc `ln F` varie vite près du zéro et la mise à jour de Broyden ne tient pas (0 %).
4. Une graine, un jeu de données par taille ; Adam à 0 % traduit surtout un mauvais réglage.
