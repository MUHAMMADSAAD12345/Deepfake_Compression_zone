import subprocess, sys, os

scripts_dir = os.path.dirname(__file__)
venv_python = os.path.join(scripts_dir, "..", "venv", "Scripts", "python.exe")

fer_cmd = [
    venv_python, os.path.join(scripts_dir, "train_fer.py"),
    "--data", os.path.join(scripts_dir, "..", "data", "fer2013"),
    "--output", os.path.join(scripts_dir, "..", "checkpoints", "fer2013_vgg19.pt"),
    "--epochs", "250", "--batch_size", "128",
]
ser_cmd = [
    venv_python, os.path.join(scripts_dir, "train_ser.py"),
    "--data", os.path.join(scripts_dir, "..", "data", "ravdess"),
    "--output", os.path.join(scripts_dir, "..", "checkpoints", "ser_weights.pt"),
    "--epochs", "500", "--batch_size", "256",
]

log_dir = os.path.join(scripts_dir, "..", "data")

procs = []
for name, cmd, log_pref in [("FER", fer_cmd, "fer"), ("SER", ser_cmd, "ser")]:
    out_f = open(os.path.join(log_dir, f"{log_pref}_gpu_out.log"), "w")
    err_f = open(os.path.join(log_dir, f"{log_pref}_gpu_err.log"), "w")
    p = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, creationflags=subprocess.CREATE_NO_WINDOW)
    procs.append((name, p, out_f, err_f))
    print(f"{name} training started (PID {p.pid})")

print("Launcher done - processes running independently on GPU")
