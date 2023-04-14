with open("inputs_ohe.txt", "r") as f:
    lines_ohe = f.readlines()

lines_ohe = [line.split() for line in lines_ohe]

with open("inputs_esm.txt", "r") as f:
    lines_esm = f.readlines()

lines_esm = [line.split() for line in lines_esm]

full_commands_ohe = [f"python train_all.py --split {line[1]} --model {line[2]} --representation {line[3]} --uncertainty {line[4]} --dropout {line[5]} --scale --seed {line[6]}" for line in lines_ohe]

full_commands_esm = [f"python train_all.py --split {line[1]} --model {line[2]} --representation {line[3]} --uncertainty {line[4]} --dropout {line[5]} --scale --mean --seed {line[6]}" for line in lines_esm]

full_commands = full_commands_ohe + full_commands_esm

with open("train_all_commands_series.sh", "w") as f:
    for line in full_commands:
        f.write(line + "\n")
