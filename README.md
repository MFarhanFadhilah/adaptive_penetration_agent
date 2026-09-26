# Adaptive Pentesting Agent (APA)

The multi-agent CTF harness that everything else builds on. It runs the agent
(a planner and two executors) that solves CTF challenges inside a Docker
sandbox, with a pluggable RAG layer for injecting knowledge hints.

- **Core** — `nyuctf_multiagent/`: planner + executor agents, LLM backends,
  tools, conversation window, Docker environment.
- **Memory (MongoDB)** — `mongo/`: Atlas Vector Search (AutoEmbed), an
  outcome-weighted ranking on top of it, an episodic log, and a self-tuning
  policy. Wired in via `--rag-mode mongo_rag`.
- **Demo** — a `webapp/` and the presentation material (not in this repo yet).

## What's here

| Path | Purpose |
|---|---|
| `nyuctf_multiagent/` | The agent framework: planner + executor agents, LLM backends, tools, conversation window, Docker environment |
| `run_rag.py` | Entry point that wires up and runs one challenge with RAG (self_rag / graph_rag / mongo_rag / combined) |
| `mongo/` | MongoDB Atlas Vector Search RAG, self-tuning policy, episodic log, and Atlas setup/health-check script |
| `configs/rag/` | Agent config (models, prompts, hyperparameters) |
| `docker/multiagent/` | The Docker image the agent runs its tools in |
| `requirements.txt`, `pyproject.toml` | Python dependencies and package metadata |

## Prerequisites

- Python 3.10+
- Docker
- An LLM API key (the config uses `gemini-3.8-flash`)
- The NYU CTF dataset (`python3 -m nyuctf.download`)
- For `--rag-mode mongo_rag`: a MongoDB Atlas cluster (`MONGODB_URI` in `keys.cfg`)

## Setup

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
docker network create ctfnet
docker build -t ctfenv:multiagent docker/multiagent
```

Create a `keys.cfg` (git-ignored) with your LLM key, for example:

```
GEMINI=your_key_here
```

## Run

```bash
python3 run_rag.py --challenge <name> --split development --max-cost 1.0
```

For Mongo-RAG, first provision Atlas:

```bash
python3 mongo/setup_atlas.py --keys keys.cfg
python3 run_rag.py --challenge <name> --split development --rag-mode mongo_rag
```

## Deploy on DigitalOcean

Runs the webapp 24/7 on a Droplet, including **Run real agent** (the agent needs
a Docker daemon, which platforms like Railway or Render don't provide).

**1. Create the Droplet.** Ubuntu 24.04 LTS, Regular/Premium Intel or AMD
(x86 — the agent image is `linux/amd64`), at least 2 vCPU / 4 GB RAM (8 GB
recommended), 50 GB disk, region close to your Atlas cluster. Add your SSH key.

**2. Allow the Droplet in Atlas.** Atlas → Network Access → add the Droplet's
public IP.

**3. Clone the repo and write `keys.cfg`** (SSH in as root):

```bash
git clone https://github.com/MFarhanFadhilah/adaptive_penetration_agent.git /opt/ctf
nano /opt/ctf/keys.cfg
```

For a private repo, clone with a GitHub personal access token as the password.
`keys.cfg`:

```
MONGODB_URI=mongodb+srv://...
OPENAI=sk-...
RUN_PASSWORD=choose-a-password
```

`RUN_PASSWORD` is asked for before each real agent run, so visitors can't spend
your OpenAI credit. Leave it out only on a private machine.

**4. Run the setup script** (installs Docker, Python deps, the dataset index,
builds the agent image, starts the webapp as a service behind Caddy, opens the
firewall; 15–25 min the first time, safe to re-run):

```bash
bash /opt/ctf/deploy/setup_droplet.sh
```

**5. Open** `http://<droplet-ip>/`. For HTTPS, point a domain at the Droplet,
replace `:80` with the domain in `/etc/caddy/Caddyfile`, then
`systemctl reload caddy`.

**Day-to-day:**

```bash
journalctl -u ctf-webapp -f                            # live logs
cd /opt/ctf && sudo -u ctf git pull && systemctl restart ctf-webapp   # deploy updates
```

The Droplet is billed while it exists, even when powered off — destroy it
after the event if you don't need it.

## Roadmap

1. **Core** — the multi-agent harness with in-memory RAG (`self_rag`,
   `graph_rag`, `combined`). Done.
2. **Memory** — the `mongo/` package and `--rag-mode mongo_rag` (MongoDB
   Atlas Vector Search + self-tuning policy), wired into `run_rag.py` and
   `rag_agent.py`. Done.
3. **Demo** — the `webapp/` and the presentation. Not started.

The core exposes a pluggable RAG interface (`RAGPlannerExecutorSystem`) so the
memory layer sits on top of it without rewriting the harness.
