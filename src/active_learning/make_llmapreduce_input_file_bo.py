command_str_list = []
i = 0
for representation in [
    # "ohe",
    "esm",
]:
    for sampling_strategy in [
        "score_greedy",
        "score_sample",
        "exploit",
        "exploit_ucb",
        "exploit_lcb",
        "exploit_ts",
    ]:
        for split in [
            # "gb1_1",
            # "gb1_2",
            # "gb1_3",
            "gb1_4",
            # "aav_2",
            # "aav_5",
            "aav_7",
            "meltome_1",
        ]:
            for model in [
                "ridge",
                "gp",
                "cnn",
            ]:
                if model == "ridge":
                    allowed_uncertainties = ["ridge"]
                elif model == "gp":
                    allowed_uncertainties = ["gp"]
                elif model == "cnn":
                    allowed_uncertainties = [
                        # "dropout",
                        "ensemble",
                        # "mve",
                        "evidential",
                        # "svi",
                    ]
                for uncertainty in allowed_uncertainties:
                    if uncertainty == "dropout":
                        for frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
                            command_str = f"{i:03} {split} {model} {representation} {uncertainty} {sampling_strategy} {frac}"
                            command_str_list.append(command_str)
                            i += 1
                    else:
                        command_str = f"{i:03} {split} {model} {representation} {uncertainty} {sampling_strategy} 0.0"
                        command_str_list.append(command_str)
                        i += 1

with open("inputs_bo.txt", "w") as f:
    for i, command_str in enumerate(command_str_list):
        if i == 0:
            f.write(command_str)
        else:
            f.write("\n" + command_str)
