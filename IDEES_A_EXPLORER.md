# Idées à explorer pour aller plus loin

L'étude montre ce que la couleur sait faire, mais pas encore une situation où elle serait plus **utile** que les méthodes classiques qui reçoivent la même information (partie 6). Les pistes ci-dessous sont classées par ce qu'elles pourraient changer à cette conclusion.

## 1. Les cas où l'échelle manque : le seul avantage plausible

La sonde n'utilise que des **rapports** de lectures (`F(z + h)/F(z)`) ; les méthodes classiques ont besoin de la valeur absolue de `|F|`. Dans la partie 6, elles l'avaient. Un avantage n'est donc à attendre que là où cette échelle manque :

- **gain du détecteur inconnu et dérivant** d'un pas à l'autre (instrument qui chauffe, amplificateur non calibré) : les rapports restent justes, les différences finies non ;
- **détecteur qui sature ou se replie** (lecture modulo une plage, exactement comme la luminosité qui se lit à une décade près) ;
- **mesure de phase seule**, sans module : la sonde unique pour une fonction holomorphe n'a besoin que de la teinte (Cauchy-Riemann).

Si la sonde ne gagne pas là non plus, la conclusion « pas d'avantage pratique » devient nette, et c'est un résultat en soi.

## 2. Résister au bruit

C'est la faiblesse mesurée (parties 4.6 et 6). Pistes :
- relire tout `W` quand la mise à jour de Broyden et la mesure divergent ;
- moyenner plusieurs sondes, ou sonder plus loin (un grand pas de différence finie sauve Newton-FD dans la partie 6) ;
- estimer le plancher de cible depuis les mesures (méthode ECnoise de Moré et Wild) au lieu de le régler à la main.

## 3. Ce que la teinte peut encore porter

La partie 5 met la courbure et la perte de validation dans la teinte. D'autres informations gratuites dans le même passage avant :
- une **contrainte** : `F = |ℓ − m0|·exp(i·violation)`, la teinte signale qu'on sort du domaine admissible ;
- un **second objectif** (compromis précision / régularité) lu comme un angle ;
- une **incertitude** (désaccord d'un ensemble de modèles), pour ralentir là où le modèle doute.

## 4. Sauver le jacobien lu en couleur

Dans [`fausses-pistes/reseau_poids_complexes/`](fausses-pistes/reseau_poids_complexes/), la couleur échoue sur des moindres carrés parce qu'elle lit des variations relatives. Idée non testée : lire `ln rᵢ` et résoudre un problème **en variables relatives**, à la manière de `W h = −1` (partie 3), en pondérant chaque résidu par sa taille connue au point courant. Si cela marche, l'holomorphie rendrait son facteur 2 aussi en couleur.

## 5. Un vrai capteur

Remplacer le capteur simulé par une **caméra qui filme un écran** affichant le domain coloring : bruit, gamma, compression et balance des blancs réels. C'est une démonstration amusante, et un vrai test de la robustesse à 8 bits.

## 6. Plusieurs zéros à la fois

La règle du produit `(fg)'/(fg) = f'/f + g'/g` permet la déflation directement sur les lectures (partie 1). Une version simultanée, à la manière d'Aberth-Ehrlich, chercherait tous les zéros d'un polynôme avec des sondes partagées.

## 7. Solidité des chiffres

- 5 graines et un réglage symétrique (même effort de réglage pour la sonde et les concurrents) dans les parties 3 à 6 ;
- dimension 32 et au-delà ;
- une revue de littérature systématique (traitement du signal, calibration d'instruments, optimisation par comparaisons) avant toute revendication de nouveauté.

## 8. L'interface

Porter le pas de Newton, la sonde à deux directions et les bits adaptatifs dans l'interface Tkinter (`src/color_descent_gui.py`), pour voir les descentes de la partie 2 en direct.
