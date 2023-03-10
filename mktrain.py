from train import get_option_interactive
from os import path, chmod
import questionary as Q


script_name = Q.text("Script name:").ask()
options = get_option_interactive()
script = """
python train.py -c {model_config} \\
    --train-data {train_data} \\
    --val-data {val_data} \\
    --total-steps {total_steps} \\
    --batch-size {batch_size} \\
    --num-workers {num_workers} \\
    --validate-every {validate_every}
    --learning-rate {learning_rate}
""".format_map(options)
script_file = path.join("scripts/", script_name)
if not script_file.endswith(".sh"):
    script_file = f"{script_file}.sh"

with open(script_file, "w") as f:
    f.write("")
chmod(script_file, 0o755)
with open(script_file, "w", encoding="utf-8") as f:
    f.write(script)
print(f"Script written to {script_file}:")
print(script)
