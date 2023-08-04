with open("inputs_bo_single.txt", "r") as f:
    lines = f.readlines()

lines = [line.split() for line in lines]

full_commands = [f"python active_learning.py --split {line[1]} --model {line[2]} --representation {line[3]} --uncertainty {line[4]} --scale --num_folds 3 --al_strategy {line[5]} --al_topk 1 --mean --dropout {line[6]} --al_step_scale single" for line in lines]

with open("bo_commands_series_single.sh", "w") as f:
    for line in full_commands:
        f.write(line + "\n")
