# Remote training on a GPU box (without using Jupyter notebooks)

You got a **JupyterLab link** to a GPU machine but want a proper terminal / VS Code
workflow. You can — the notebook UI is optional. Pick one of the three setups below,
then run the same commands.

## Get a real shell — three options

### Option A — JupyterLab built-in terminal (works immediately, zero setup)
1. Open the Jupyter link in the browser → you land in **JupyterLab**.
2. **File → New → Terminal** (or the "Terminal" tile in the Launcher).
3. That's a full `bash` shell **on the GPU machine**. You never touch a notebook.

### Option B — VS Code Remote-SSH (best experience, needs SSH access)
Most GPU providers that hand out a Jupyter link *also* expose SSH (check the dashboard
for a host / port / key — e.g. `ssh -p 12345 root@1.2.3.4`).
1. Install the **Remote - SSH** extension in VS Code.
2. `Cmd/Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → paste the SSH string.
3. Open the folder, use the integrated terminal — full local-like experience.

### Option C — VS Code in the browser (if no SSH, but you can install things)
In the Jupyter terminal: `pip install jupyter-vscode-proxy code-server` then launch
`code-server` — opens VS Code in a browser tab. (Option A is simpler; use this only if
you want the VS Code editor specifically.)

> If SSH is available, use **Option B**. If you only have the Jupyter link, **Option A**
> is the fastest and is all you actually need.

## Then: get the code + data + run (same for all options)

In the shell on the GPU box:

```bash
# 1) Code — clone the repo (private, so use a GitHub token or `gh auth login`)
git clone https://github.com/Luraxx/msd-floorplan-challenge.git
cd msd-floorplan-challenge

# 2) Deps + dataset (needs a Kaggle API token at ~/.kaggle/kaggle.json)
bash scripts/setup_remote.sh

# 3) Train at full scale on the GPU, then predict + eval
bash scripts/train_full.sh
# or tune: SIZE=512 EPOCHS=150 BATCH=32 bash scripts/train_full.sh
```

The model auto-detects the GPU (`cuda` → `mps` → `cpu`), so on the GPU box it uses CUDA
with no changes.

## The two things that actually need credentials

1. **GitHub** (private repo): create a Personal Access Token (github.com → Settings →
   Developer settings → Tokens) and clone with
   `git clone https://<TOKEN>@github.com/Luraxx/msd-floorplan-challenge.git`,
   or run `gh auth login` first.
2. **Kaggle** (dataset, 16 GB): kaggle.com → Account → *Create New API Token* → upload the
   downloaded `kaggle.json` to `~/.kaggle/kaggle.json` on the box, then
   `chmod 600 ~/.kaggle/kaggle.json`. (Alternative source: the 4TU dataset page.)

## Getting results back

The run prints FID / density / coverage at the end. To keep artifacts, copy them down:
```bash
# weights + predictions live in outputs/
#  - outputs/unet.pt
#  - outputs/generated_unet/*.pickle
```
With SSH: `scp -P <port> root@<host>:msd-floorplan-challenge/outputs/unet.pt .`
With Jupyter only: right-click the file in the JupyterLab file browser → **Download**.

## Scaling knobs (env vars for `train_full.sh`)

| Var | Default | Effect |
|---|---|---|
| `SIZE` | 256 | raster resolution — higher = sharper rooms, better FID, more VRAM |
| `EPOCHS` | 100 | training length |
| `BATCH` | 32 | batch size — raise to fill the GPU |
| `NTRAIN` | 4572 | training samples (full split) |
| `NTEST` | 800 | test samples to score |

Local baseline for reference (small, MPS): size 128, n 1500, 25 epochs → FID ~150.
Retrieval baselines (for comparison): v1 FID 36, v2 FID 34.
