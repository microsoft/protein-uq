python active_learning.py --split gb1_4 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy score_sample --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_ucb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_lcb --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split gb1_4 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split aav_7 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model gp --representation esm --uncertainty gp --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty ensemble --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
python active_learning.py --split meltome_1 --model cnn --representation esm --uncertainty evidential --scale --num_folds 3 --al_strategy exploit_ts --num_al_loops 5 --al_topk 100 --mean --dropout 0.0