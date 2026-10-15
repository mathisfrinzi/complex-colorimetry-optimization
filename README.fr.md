# Optimiser avec des couleurs

🇬🇧 [English version](README.md) · 🇫🇷 Français

*complex-colorimetry-optimization*

**Peut-on trouver les zéros d'une fonction, ou minimiser une perte, en ne regardant que les couleurs de son *domain coloring* ?**

La règle du jeu : l'algorithme ne lit jamais la valeur de `f`. Il ne voit qu'une couleur, quantifiée sur 8 ou 12 bits par canal. La **teinte** donne l'argument de `f`, la **luminosité** donne `log|f|` à une décade près. On part de l'algorithme le plus naïf (« aller vers le plus sombre »), on l'améliore limite après limite, et à chaque étape on le compare aux méthodes classiques (Newton, BFGS, Nelder-Mead, CMA-ES, Adam…) à **nombre d'évaluations égal**.

**Réponse courte.**
- **Pour trouver des zéros, ça marche très bien.** Deux lectures de couleur suffisent à faire un pas de Newton complet, car les modules s'éliminent. Avec un capteur 8 bits, on trouve les zéros aussi souvent que les meilleures méthodes classiques, et souvent pour moins d'évaluations.
- **Pour minimiser une perte ordinaire, il faut deux idées de plus** : une cible qu'on abaisse en route, puis une courbure apprise. La méthode atteint alors le taux de succès de BFGS jusqu'en dimension 20, pour 1,2 à 2 fois son coût.
- **La teinte peut porter une deuxième information gratuite.** Sur un petit réseau, elle lit la courbure (−30 à −52 % d'évaluations) ou la perte de validation (arrêt précoce lu dans la couleur).
- **À armes égales, pas d'avantage.** Quand les méthodes classiques reçoivent exactement les mêmes mesures dégradées, Broyden classique reste moins cher, et la sonde est plus fragile au bruit.

## L'idée en quatre images

**1. Le capteur.** Chaque point reçoit une couleur. La teinte donne `arg f`, la luminosité fait une dent de scie à chaque décade de `|f|`. Le capteur ne donne jamais la valeur de `f`.

![Le capteur](schemas/s1_capteur.png)

**2. Pourquoi la couleur montre le chemin.** Pour une fonction holomorphe, les lignes de même teinte croisent à angle droit les lignes de même `|f|` (Cauchy-Riemann). Descendre `|f|`, c'est suivre une ligne de teinte constante jusqu'au zéro.

![Cauchy-Riemann](schemas/s3_cauchy_riemann.png)

**3. De l'anneau à la sonde.** L'algorithme naïf lit 9 couleurs autour du point et avance vers la plus sombre. Or la différence de couleur entre **deux** points donne directement `L = ln f(z+η) − ln f(z) ≈ η·f'/f` : la variation de luminosité en est la partie réelle, la variation de teinte la partie imaginaire. On en tire la direction **et** la longueur du pas de Newton, `h = −f/f'`.

![Anneau contre sonde](schemas/s2_anneau_vs_sonde.png)

**4. Une vraie descente** avec le capteur 8 bits : chaque pas coûte deux lectures et divise `|f|` par un facteur à peu près constant.

![Une descente](schemas/s7_une_descente.png)

## Le fil de l'étude

Chaque étape est née d'une limite mesurée de la précédente.

![Le fil de l'étude](schemas/s13_fil_de_l_etude.png)

## Les résultats en un graphe

Pour chaque situation, la meilleure méthode couleur face à la meilleure référence classique. À gauche le succès, à droite le coût relatif.

![Bilan](schemas/s14_bilan.png)

À lire avec une réserve : sauf pour les lignes marquées ★, la référence lit la **valeur exacte** de `f` en double précision, alors que la sonde lit un capteur quantifié. La partie 6 est la seule comparaison où tout le monde lit les mêmes mesures.

| situation | verdict | chiffres clés | partie |
|---|---|---|---|
| zéros de fonctions holomorphes (C → C) | **très bien** | 99-100 % en 24-35 évaluations ; Nelder-Mead 99 % en 50, BFGS 99 % en 43 | [2](parties/2_zeros_en_2d/) |
| zéros de fonctions non holomorphes (sonde à 2 directions) | **très bien** | 99,8-100 % en 21-30 évaluations | [2](parties/2_zeros_en_2d/) |
| vallées étroites, mauvais conditionnement (pas de Newton lu en couleur) | bien | Rosenbrock 0 % → 79-99 % ; fonctions dures 80 % (Nelder-Mead 95 %) | [2](parties/2_zeros_en_2d/) |
| capteur grossier, changement d'échelle | bien | 91-97 % dès 6 bits ; invariant d'échelle de 10⁻¹² à 10¹² | [2](parties/2_zeros_en_2d/) |
| systèmes de n équations (sonde-Broyden) | bien | coût lent en n ; bat Newton-FD, LM et BFGS dès n ≈ 8, **pas** Broyden classique | [3](parties/3_systemes_dimension_n/) |
| boîte noire, face aux méthodes sans dérivée générales | gagne | système linéaire, n = 16 : 100 % en 48 lectures ; Nelder-Mead, Powell, CMA-ES : 0 % | [3](parties/3_systemes_dimension_n/) |
| perte à minimum non nul | échoue, puis **corrigé** | sonde de base 9 % → cible 69 % → cible + courbure 81 % (BFGS 82 %) ; n = 20 : 81 % (BFGS 75 %) pour 1,2-2 fois son coût | [4](parties/4_pertes_minimum_non_nul/) |
| perte bruitée | limité | la plus robuste des méthodes à gradient (54 % contre 13 % pour BFGS à 0,1 % de bruit), loin derrière CMA-ES (87 %) | [4](parties/4_pertes_minimum_non_nul/) |
| petit réseau, la teinte lit la courbure | bien en boîte noire | 85 % (BFGS par différences finies : 84 %), 30-52 % d'évaluations en moins ; BFGS avec rétropropagation reste 6-18 fois moins cher | [5](parties/5_reseau_de_neurones/) |
| la teinte lit la perte de validation | **réussi** | arrêt précoce lu dans la couleur = arrêt avec les vraies valeurs (validation 0,0197 contre 0,0368 sans arrêt) | [5](parties/5_reseau_de_neurones/) |
| fonctions multimodales (Rastrigin) | échoue | tombe dans les creux, comme toute méthode de descente | [2](parties/2_zeros_en_2d/), [4](parties/4_pertes_minimum_non_nul/) |
| **mêmes mesures dégradées pour tous** | pas d'avantage | Broyden classique moins cher ; sonde fragile dès 2 % de bruit pour n ≥ 8 | [6](parties/6_a_armes_egales/) |

## Ce que la teinte apporte en plus

Tant qu'on minimise une perte réelle, la teinte ne sert à rien : on lit la couleur d'un nombre réel. Mais une lecture de couleur donne un **rapport complexe**, donc la variation de deux quantités pour le prix d'une. En lisant `F = |ℓ − m0|·exp(−iλD)`, où `D` est l'écart des prédictions du réseau par rapport au point courant, la luminosité lit la pente et la teinte lit la courbure de Gauss-Newton dans la direction sondée :

![La teinte lit la courbure](schemas/s12_teinte_lit_la_courbure.png)

Avec `F = ℓ_entraînement·exp(i·ln ℓ_validation)`, la teinte suit la perte de validation, et l'arrêt précoce peut se décider en regardant la couleur :

![La teinte montre le sur-apprentissage](parties/5_reseau_de_neurones/figures/validation_teinte.png)

## Ce qui est nouveau, et ce qui ne l'est pas

Chaque brique est connue : Newton sur `ln f`, la méthode de Broyden, le pas de Polyak avec une valeur cible (1969), BFGS, le réglage des différences finies au bruit. En suivant la couleur, l'étude les **retrouve** une à une. Ce qui m'a semblé propre à ce travail :
- la couleur comme seul capteur, et la mesure de ce qu'elle coûte face aux méthodes classiques ;
- la cible `m0`, qui sert non seulement à régler le pas mais surtout à **rendre le signal lisible** près du minimum ;
- la teinte comme second canal (courbure, validation) dans la même lecture que la pente.

Références et positionnement : [`docs/REFERENCES.md`](docs/REFERENCES.md).

## Parcours

1. [**Le capteur et la sonde**](parties/1_capteur_et_sonde/) : le capteur, l'interface d'exploration, les anneaux, les théorèmes (Cauchy-Riemann, principe de l'argument, minimum du module), puis la sonde à deux lectures.
2. [**Zéros en 2D**](parties/2_zeros_en_2d/) : chaque limite donne une amélioration (plateaux → règle aveugle, non-holomorphie → deux directions, vallées → pas de Newton, coût → sonde adaptative et bits adaptatifs). Banc principal sur 5 graines.
3. [**Systèmes en dimension n**](parties/3_systemes_dimension_n/) : `W h = −1`, réutiliser les lectures (sonde-Broyden), bits adaptatifs, et la comparaison en boîte noire.
4. [**Pertes à minimum non nul**](parties/4_pertes_minimum_non_nul/) : la sonde de base échoue ; cible, courbure, diagnostic du coût, dimension n, bruit.
5. [**Réseau de neurones**](parties/5_reseau_de_neurones/) : la sonde à cible sur un petit réseau, puis la teinte au travail (courbure, validation).
6. [**À armes égales**](parties/6_a_armes_egales/) : un problème d'annulation avec un détecteur gain/phase, mêmes mesures pour toutes les méthodes.

À côté : [**fausses pistes**](fausses-pistes/) (ce qui n'a pas marché, et pourquoi) et [**idées à explorer**](IDEES_A_EXPLORER.md).

## Structure du dépôt

```
README.fr.md / README.md   cette page (français) / version anglaise
IDEES_A_EXPLORER.md        pistes pour aller plus loin
docs/REFERENCES.md         références et positionnement
schemas/                   14 schémas (make_schemas.py, make_syntheses.py)
src/                       code partagé : capteur, sondes, interface Tkinter
parties/
  1_capteur_et_sonde/      texte, captures de l'interface, notebooks, tests sur tanh
  2_zeros_en_2d/           bancs 2D (comparaison, non holomorphes, dures, final sur 5 graines, sensibilité, bits)
  3_systemes_dimension_n/  bancs en Cⁿ et en boîte noire
  4_pertes_minimum_non_nul/ sonde à cible, courbure, coût, dimension n, bruit
  5_reseau_de_neurones/    petit réseau élève-professeur, teinte = courbure, teinte = validation
  6_a_armes_egales/        détecteur gain/phase, mêmes mesures pour tous
fausses-pistes/            essais écartés, avec leurs scripts et résultats
```

Dans chaque partie, `bench_*.py` lance un banc et écrit `*_results.json` ; `bench_*_figs.py` en tire les tableaux (`*_tables.md`) et les figures (`figures/`).

## Lancer

```
pip install -r requirements.txt
python src/color_descent_gui.py                    # interface d'exploration (Tkinter)
python parties/2_zeros_en_2d/bench2d.py --smoke    # vérification rapide d'un banc
python schemas/make_schemas.py                     # régénère les schémas 1 à 11
python schemas/make_syntheses.py                   # régénère les schémas 12 à 14
sh run_long.sh                                     # tout recalculer (≈ 2 h)
```

## Limites

- Cinq graines de départs pour la partie 2, une seule pour les parties 3 à 6.
- Fonctions de test choisies par moi : elles couvrent des cas classiques, pas de vrais problèmes industriels.
- Hors partie 6, les méthodes classiques lisent la valeur exacte de `f` : la comparaison leur est favorable sur ce point, mais elle ne dit pas ce qu'elles feraient avec un capteur aussi pauvre.
- Revue de littérature non systématique.

## Licence

Code : [MIT](LICENSE) (code under MIT). Textes, schémas et figures : [CC BY 4.0](LICENSE-CONTENT.md). © Mathis Frinzi, [github.com/mathisfrinzi](https://github.com/mathisfrinzi).
