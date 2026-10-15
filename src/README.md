# Code partagé

| module | contenu | utilisé dans |
|---|---|---|
| `color_descent.py` | le capteur `color_of(F, bits)` (teinte = arg, luminosité = bande de log₁₀\|F\|) et les **anneaux** de luminosité et de teinte | partie 1 |
| `color_probe.py` | lecture `sense`, **sonde unique** avec règle « aveugle » et déflation | parties 1, 2 |
| `color_probe2.py` | sonde à **deux directions** (fonctions non holomorphes), pas de plus grande pente | partie 2 |
| `color_probe3.py` | sonde « v3 » : 1, 2 ou nombre adaptatif de directions, **pas de Newton** tiré des couleurs, arrêt sans oracle, **bits adaptatifs** (`bits_adapt`) | parties 2, schémas |
| `color_probe_cn.py` | sonde en dimension n : `descend_cn` (relit W à chaque pas) et `descend_cn_broyden` (1 lecture par pas, bits adaptatifs) | parties 3, 6 |
| `color_descent_gui.py` | interface Tkinter (anneaux, sonde, règle aveugle ; pas encore le pas de Newton) | partie 1 |
| `sonde_cible.py` | **sonde à cible + courbure** en dimension n, pour minimiser une perte à minimum non nul (cible m0, BFGS, gradient lu en couleur) ; `sonde_cible_teinte` : la teinte lit la courbure | parties 4, 5 |
| `sonde_residus.py` | sonde « résidus » : Levenberg-Marquardt dont le jacobien est lu sur les couleurs des résidus (une couleur par donnée), avec ou sans holomorphie | fausse piste (poids complexes) |
