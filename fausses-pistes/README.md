# Fausses pistes

Ce qui n'a pas marché, et pourquoi. Ces essais n'apportent rien au fil principal, mais ils expliquent plusieurs choix des parties 1 à 6 et évitent de refaire les mêmes erreurs.

## Deux essais complets

| dossier | idée | résultat | ce qu'on en a tiré |
|---|---|---|---|
| [`reseau_deux_pertes_redondantes/`](reseau_deux_pertes_redondantes/) | entraîner un réseau en cherchant un zéro de `F = L_A + i·L_B`, deux pertes sur deux moitiés des données | sonde 7 à 43 % de succès, Broyden 0 % | la partie imaginaire doit porter une information **différente** de la perte : d'où la courbure et la validation de la partie 5 |
| [`reseau_poids_complexes/`](reseau_poids_complexes/) | poids complexes holomorphes : une sonde par paramètre complexe au lieu de deux | holomorphie : coût divisé par 1,7 à 1,9 avec un jacobien exact ; jacobien lu en couleur : 0 à 10 % de succès | la couleur lit des variations **relatives** ; elle convient au pas de Newton `W h = −1`, pas aux moindres carrés à résidus non nuls |

## Petits essais écartés en route

| partie | essai | résultat | pourquoi |
|---|---|---|---|
| 2.2 | sur le plateau de tanh, tirer une direction de sonde au hasard à chaque pas | 52 % (règle « aveugle » : 89 %) | une direction aléatoire perd l'information des pas précédents |
| 2.2 | orienter la sonde sur la dernière direction de descente | 40 % | sur le plateau, la dernière direction est justement celle qui ne voit rien |
| 3.3 | bits adaptatifs ajoutés seulement quand la sonde est aveugle | 10 % à n = 8 | à 5 bits la matrice `W` devient singulière en boucle ; corrigé en ajoutant aussi des bits dans ce cas |
| 4.3 | gradient lu par différence avant (2 lectures) pour construire la courbure BFGS en 2D | 59-68 %, moins bien que sans courbure (69-70 %) | le biais de la différence avant fausse les *différences* de gradients dont BFGS tire la courbure ; la différence centrée le supprime (79-80 %) |
| 4.4 | différence avant corrigée par la courbure apprise | succès 90 % → 81 % des départs | la courbure apprise n'est pas assez précise pour corriger le biais |
| 4.4 | régler η sur le gradient prédit au nouveau point | 46 % | η trop grand, la sonde enjambe le niveau `m0` |
| 4.5 | lire quelques directions du gradient et prédire les autres avec le modèle BFGS | 6-38 % à n = 5-20 | même raison : la courbure apprise ne prédit pas un gradient |
| 5 | critère de succès : atteindre le meilleur minimum de 40 départs de BFGS | 5 % pour **toutes** les méthodes à 24 paramètres | ce minimum apprend le bruit ; on vise désormais la perte du professeur |
| 5.2 | régler λ sur la courbure médiane | 45 % | la direction la plus courbée fait déborder la teinte (jusqu'à 200 relectures par course) |
| 5.2 | utiliser la courbure lue direction par direction (hessienne diagonale) | 74-77 % contre 82-85 % | les paramètres d'un réseau sont couplés ; on ne garde que l'échelle de la courbure |
| 5.2 | forme polaire sans garder `λη²` fixe | jusqu'à 1 000 relectures par course | `D` varie comme `η²` : la teinte fait plus d'un demi-tour |
| 6 | garder pour les méthodes classiques le pas de différence finie de 10⁻⁶ des parties précédentes, alors que les mesures sont quantifiées | 0 % | la quantification noie la différence ; avec un pas réglé, Newton-FD, Broyden et LM redeviennent excellents |

Ces fichiers se lancent comme les autres bancs (`python fausses-pistes/<dossier>/<script>.py`), depuis la racine du dépôt.
