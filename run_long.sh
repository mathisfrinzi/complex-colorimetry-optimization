#!/bin/sh
# Relance de tous les calculs (≈ 2 h sur 2 cœurs). À lancer depuis la racine du dépôt.
# Les résultats sont déjà dans parties/*/ ; ce script les régénère, puis refait tableaux, figures et schémas.
set -e
pip install -r requirements.txt
P=parties
python $P/2_zeros_en_2d/bench2d.py          # ≈ 3 min
python $P/2_zeros_en_2d/bench_nonholo.py    # ≈ 3 min
python $P/2_zeros_en_2d/bench_hard.py       # ≈ 7 min
python $P/2_zeros_en_2d/bench_v3.py         # ≈ 25-45 min (5 graines)
python $P/2_zeros_en_2d/bench_sens.py       # ≈ 30-40 min
python $P/2_zeros_en_2d/bench_bits.py       # ≈ 5 min (après bench_v3)
python $P/3_systemes_dimension_n/bench_cn.py             # ≈ 12 min
python $P/3_systemes_dimension_n/bench_cn_simple.py      # ≈ 10 min
python $P/3_systemes_dimension_n/bench_cn_bits.py        # ≈ 1 min
python $P/4_pertes_minimum_non_nul/bench_loss.py     # ≈ 4 min
python $P/4_pertes_minimum_non_nul/bench_loss_cible.py  # ≈ 2 min (sonde à cible)
python $P/4_pertes_minimum_non_nul/bench_loss_cible_qn.py  # ≈ 2 min (sonde à cible + courbure)
python $P/4_pertes_minimum_non_nul/diag_cout.py        # ≈ 1 min (où partent les évaluations)
python $P/4_pertes_minimum_non_nul/bench_loss_nd.py    # ≈ 7 min (dimension n)
python $P/4_pertes_minimum_non_nul/bench_loss_bruit.py # ≈ 17 min (bruit)
python $P/4_pertes_minimum_non_nul/bench_loss_bruit.py sonde-cible-plancher-12b  # ≈ 11 min (bruit, plancher)
python $P/3_systemes_dimension_n/bench_blackbox.py       # ≈ 2 min (après bench_cn_simple)
python $P/5_reseau_de_neurones/bench_net_cible.py           # ≈ 4 min (5.1)
python $P/5_reseau_de_neurones/bench_net_teinte.py          # ≈ 3 min (5.2)
python $P/5_reseau_de_neurones/bench_net_validation.py      # ≈ 4 min (5.3, 5.4)
python $P/6_a_armes_egales/bench_usecase.py          # ≈ 12 min
for f in $P/*/*_figs.py; do python $f > /dev/null; done
python schemas/make_schemas.py
python schemas/make_syntheses.py
# fausses pistes (facultatif)
python fausses-pistes/reseau_deux_pertes_redondantes/bench_net.py   # ≈ 9 min
python fausses-pistes/reseau_poids_complexes/bench_net_holo.py      # ≈ 2 min
