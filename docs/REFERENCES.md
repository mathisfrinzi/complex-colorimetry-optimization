# Références et positionnement

*Une douzaine de recherches ciblées, pas une revue systématique.*

## En bref

Chaque brique de l'étude existe déjà dans la littérature : le domain coloring, le principe de l'argument, le pas de Newton, Broyden et BFGS, le pas de Polyak avec une valeur cible qu'on ajuste, les différences finies adaptées au bruit. **Ce qu'on n'a pas trouvé** : un algorithme qui utilise la *couleur* d'un domain coloring (quantifiée, luminosité lue à une décade près) comme seul capteur pour trouver des zéros ou minimiser une perte. L'apport de l'étude est donc la **combinaison** de ces briques sous cette contrainte de capteur, et la mesure de ce qu'elle vaut. Ce n'est pas une méthode nouvelle au sens fort.

## Ce que l'étude retrouve, partie par partie

### Domain coloring et portraits de phase (partie 1)
- E. Wegert, *Visual Complex Functions: An Introduction with Phase Portraits*, Birkhäuser, 2012. [Springer](https://link.springer.com/doi/10.1007/978-3-0348-0180-5)
- E. Wegert, « Visual Exploration of Complex Functions », chapitre, 2016. [PDF](https://math.okstate.edu/people/scurry/5283/sp21/17_Wegert2016_Chapter_VisualExplorationOfComplexFunctions.pdf)
- [Domain coloring (Wikipédia)](https://en.wikipedia.org/wiki/Domain_coloring)

### Compter et localiser les zéros par la phase (partie 1)
- L. M. Delves, J. N. Lyness, « A numerical method for locating the zeros of an analytic function », *Mathematics of Computation* 21 (1967). C'est la référence du comptage de zéros par le principe de l'argument, que l'étude refait avec la teinte. Des travaux récents le prolongent, par exemple [*Computing the Zeros of a Holomorphic Function Using Quadrature-Based Subdivision…* (arXiv 2509.15936)](https://arxiv.org/pdf/2509.15936) et [*On locating the zeros and poles of a meromorphic function* (arXiv 2103.09047)](https://arxiv.org/pdf/2103.09047).

### Descente de |f| et flot de Newton (parties 1 et 2)
- H. Th. Jongen, P. Jonker, F. Twilt, « The continuous, desingularized Newton method for meromorphic functions », *Acta Applicandae Mathematicae* (1988). [PDF](https://ris.utwente.nl/ws/files/18474443/Jongen_10.1007_BF00047503.pdf) Les lignes de teinte constante sont les trajectoires du flot de Newton.
- Le pas `−1/A = −f/f'` tiré de deux couleurs est le pas de Newton, appliqué à `ln f` : `A` est une différence finie de la dérivée logarithmique.

### Systèmes et réutilisation des lectures (partie 3)
- C. G. Broyden, « A class of methods for solving nonlinear simultaneous equations », *Mathematics of Computation* 19 (1965). [DOI](https://www.doi.org/10.1090/S0025-5718-1965-0198670-6) La sonde-Broyden de la partie 3 est la méthode de Broyden appliquée à la matrice `W = ∂ ln Fᵢ/∂zⱼ` lue en couleur.

### Sonde à cible : le pas de Polyak avec une valeur cible (partie 4)
- B. T. Polyak, « Minimization of nonsmooth functionals », *Zh. Vychisl. Mat. Mat. Fiz.* 9 (3), 1969, p. 509-521. [Math-Net.ru](https://www.mathnet.ru/eng/zvmmf7159) Le pas `h = −(ℓ − m0)·∇ℓ/|∇ℓ|²` de la partie 4.2 est exactement le pas de Polyak avec la valeur cible `m0`.
- E. Hazan, S. Kakade, *Revisiting the Polyak step size*, 2019. [arXiv 1905.00313](https://arxiv.org/pdf/1905.00313)
- N. Loizou, S. Vaswani, I. Laradji, S. Lacoste-Julien, « Stochastic Polyak Step-size for SGD: An Adaptive Learning Rate for Fast Convergence », AISTATS 2021. [PMLR](https://proceedings.mlr.press/v130/loizou21a.html)
- A. Liu, M. A. Bragin, X. Chen, X. Guan, *Accelerating Level-Value Adjustment for the Polyak Stepsize*. [arXiv 2311.18255](https://arxiv.org/html/2311.18255v3) Ce travail ajuste la valeur cible en route, ce qui est l'idée « m0 ← m0 − ε » de la partie 4.2. Il la *remonte* quand le pas est trop grand ; la sonde à cible la place à la baisse obtenue et la divise par 2 en cas d'échec.
- **Ce qui est propre à l'étude** : lire `ln|ℓ − m0|` par un capteur logarithmique quantifié, et remarquer que la cible sert alors aussi à *amplifier le signal* près du minimum (schéma s9). Dans la littérature, la cible sert à régler la longueur du pas, pas à rendre la mesure lisible.

### Quasi-Newton par différences finies, et bruit (parties 4.3 à 4.6)
- J. Nocedal, S. J. Wright, *Numerical Optimization*, Springer, 2e éd., 2006 : BFGS, différences finies, recherche linéaire.
- A. S. Berahas, R. H. Byrd, J. Nocedal, « Derivative-Free Optimization of Noisy Functions via Quasi-Newton Methods », *SIAM J. Optim.* 2019. [Optimization Online](https://optimization-online.org/2018/03/6543/) Ils règlent le pas de différence finie à partir du niveau de bruit estimé. Le **plancher de cible** de la partie 4.6 joue le même rôle : empêcher que les différences lues tombent sous le bruit.
- J. J. Moré, S. M. Wild, « Estimating Computational Noise », *SIAM J. Sci. Comput.* 2011 (ECnoise). [Optimization Online](https://optimization-online.org/2010/03/2557/) Une piste pour régler le plancher sans réglage manuel.

### Optimisation avec une information grossière (contexte)
- *Adapting Zeroth Order Algorithms for Comparison-Based Optimization*. [arXiv 2210.05824](https://arxiv.org/pdf/2210.05824)
- *Gradient Testing and Estimation by Comparisons*. [arXiv 2405.11454](https://arxiv.org/html/2405.11454v2)
- Ces travaux étudient l'optimisation quand on ne lit que des comparaisons ou des valeurs quantifiées : c'est la famille de problèmes la plus proche de « lire une couleur ».

## Positionnement

Partant de « suivre la couleur », l'étude retrouve pas à pas Newton sur `ln f`, Broyden, le pas de Polyak avec cible, BFGS et le réglage des différences finies au bruit. Je la présente donc comme une **re-découverte guidée par une contrainte de capteur**, pas comme une méthode nouvelle.

Ce qui me semble propre à ce travail :
- utiliser la couleur d'un domain coloring (quantifiée, luminosité lue à une décade près) comme **seul capteur** ;
- la remarque que la cible `m0` sert aussi à **amplifier le signal lisible** près du minimum, et pas seulement à régler la longueur du pas ;
- la teinte utilisée comme second canal (courbure, perte de validation) dans la même lecture que la pente ;
- la comptabilité des coûts, y compris la comparaison où toutes les méthodes reçoivent les mêmes mesures dégradées (partie 6).

La revue reste partielle : une revue systématique (bases bibliographiques, mots-clés en traitement du signal et en calibration d'instruments) serait nécessaire avant toute revendication de nouveauté.

Où ces références interviennent : partie 1 (Wegert, Delves-Lyness, Jongen et al.), partie 3 (Broyden), partie 4 (Polyak, Hazan-Kakade, Loizou et al., Liu et al., Berahas et al., Moré-Wild).
