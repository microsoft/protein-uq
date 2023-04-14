command_str_list = []
i = 0
for representation in [
    "ohe",
]:
    for split in [
        "gb1_1",
        "gb1_2",
        "gb1_3",
        "gb1_4",
        "aav_2",
        "aav_5",
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
                cross_val = False
            elif model == "gp":
                allowed_uncertainties = ["gp"]
                cross_val = False
            elif model == "cnn":
                allowed_uncertainties = [
                    "dropout",
                    "ensemble",
                    "mve",
                    "evidential",
                    "svi",
                ]
                cross_val = True
            for uncertainty in allowed_uncertainties:
                if uncertainty == "dropout":
                    for frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        if cross_val:
                            for cv_index in [0, 1, 2, 3, 4]:
                                command_str = f"{i:03} {split} {model} {representation} {uncertainty} {frac} {cv_index}"
                                command_str_list.append(command_str)
                                i += 1
                        else:
                            command_str = f"{i:03} {split} {model} {representation} {uncertainty} {frac} 0"
                            command_str_list.append(command_str)
                            i += 1
                else:
                    if cross_val:
                        for cv_index in [0, 1, 2, 3, 4]:
                            command_str = f"{i:03} {split} {model} {representation} {uncertainty} 0.0 {cv_index}"
                            command_str_list.append(command_str)
                            i += 1
                    else:
                        command_str = f"{i:03} {split} {model} {representation} {uncertainty} 0.0 0"
                        command_str_list.append(command_str)
                        i += 1

with open("inputs_ohe.txt", "w") as f:
    for i, command_str in enumerate(command_str_list):
        if i == 0:
            f.write(command_str)
        else:
            f.write("\n" + command_str)
