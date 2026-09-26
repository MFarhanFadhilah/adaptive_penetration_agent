"""
Local demo website for the MongoDB-backed pentest/CTF agent memory system.

Connects to the real Atlas cluster (pentest_memory) and:
  - lists 43 real NYU CTF Bench challenges pulled from the dataset on disk
  - "simulates" a Planner-Executor episode per challenge (no Docker/LLM
    exploitation needed) while doing REAL Atlas work: vector-search
    retrieval via MongoRAG, real writes to episodic_log/runs (tagged
    simulated=True), which fire the real outcome-tracking Atlas Trigger,
    followed by a real call to the self-tuning policy update.
  - runs the REAL agent (run_rag.py, mongo_rag mode, OpenAI models) on a
    challenge and streams its console output live. Needs Docker, the
    ctfenv:multiagent image and OPENAI= in keys.cfg.
  - visualizes, round by round, the difference between resending the full
    tool-output history every step (token overload / context bloat) and
    the harness's real sliding-window + truncation approach (bounded).

Run from the llm_ctf_automation directory:
    python3 webapp/app.py
Then open http://localhost:8765
"""
import json
import os
import random
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from pymongo import MongoClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nyuctf_multiagent.utils import APIKeys  # noqa: E402
from mongo.mongo_rag import MongoRAG  # noqa: E402
from mongo.policy import POLICY_DEMO, default_policy, get_policy, update_policy  # noqa: E402

NYUCTF_ROOT = Path.home() / ".nyuctf" / "v20250206"
# How many NYU CTF development-split challenges to show per category
# (the dev split has 10 each for web/pwn/rev/forensics and 11 for crypto)
CHALLENGES_PER_CATEGORY = {"web": 10, "pwn": 10, "rev": 10, "crypto": 3, "forensics": 10}
# Real agent runs (run_rag.py) launched from the webapp
REAL_AGENT_CONFIG = "configs/rag/mongo_rag_openai_config.yaml"
REAL_AGENT_IMAGE = "ctfenv:multiagent"
REAL_AGENT_NETWORK = "ctfnet"
REAL_AGENT_MAX_COST_CAP = 5.0  # hard ceiling on the per-run budget a viewer can request
real_run_lock = threading.Lock()  # one real run at a time: each one starts Docker containers
WINDOW = 5  # matches len_observations default in nyuctf_multiagent/conversation.py
BASE_OVERHEAD_TOKENS = 480  # rough system prompt + tool schema overhead
COST_PER_TOKEN = 0.0000005  # matches the blended rate seen in existing runs data

app = Flask(__name__, static_folder="static", static_url_path="/static")

keys = APIKeys(str(REPO_ROOT / "keys.cfg"))
mongo_client = MongoClient(keys["MONGODB_URI"])
db = mongo_client["pentest_memory"]
semantic_col = db["semantic_memory"]
runs_col = db["runs"]
episodic_col = db["episodic_log"]
policy_col = db["policy"]

ai_gateway = None
if "OPENAI" in keys:
    try:
        from mongo.ai_gateway import MongoAIGateway
        # OpenAI-compatible client; the gateway appends /v1 to the endpoint
        ai_gateway = MongoAIGateway("https://api.openai.com", keys["OPENAI"])
    except Exception:
        ai_gateway = None
elif "MONGODB_AI_ENDPOINT" in keys and "MONGODB_AI_KEY" in keys:
    try:
        from mongo.ai_gateway import MongoAIGateway
        ai_gateway = MongoAIGateway(keys["MONGODB_AI_ENDPOINT"], keys["MONGODB_AI_KEY"])
    except Exception:
        ai_gateway = None


def load_challenges():
    manifest = json.loads((NYUCTF_ROOT / "development_dataset.json").read_text())
    by_category = {}
    for key in sorted(manifest.keys()):
        meta = manifest[key]
        by_category.setdefault(meta["category"], []).append((key, meta))

    chosen = []
    for cat, n in CHALLENGES_PER_CATEGORY.items():
        chosen.extend(by_category.get(cat, [])[:n])

    challenges = []
    for key, meta in chosen:
        chal_json_path = NYUCTF_ROOT / meta["path"] / "challenge.json"
        description, points = "", None
        if chal_json_path.exists():
            try:
                cj = json.loads(chal_json_path.read_text())
                description = cj.get("description", "")
                points = cj.get("points")
            except Exception:
                pass
        challenges.append({
            "id": key,
            "path": meta["path"],
            "year": meta["year"],
            "event": meta["event"],
            "category": meta["category"],
            "name": meta["challenge"],
            "description": description,
            "points": points,
        })
    return challenges


CHALLENGES = load_challenges()
CHALLENGES_BY_ID = {c["id"]: c for c in CHALLENGES}

TOOL_POOL = {
    "web": ["curl -v", "sqlmap --batch", "gobuster dir", "burpsuite repeater", "view-source"],
    "pwn": ["checksec", "gdb -q ./chall", "pwntools cyclic", "objdump -d", "ropper --search"],
    "rev": ["strings ./chall", "objdump -d", "ghidra headless", "ltrace ./chall", "radare2 -A"],
    "crypto": ["openssl s_client", "python3 factor.py", "frequency analysis", "RsaCtfTool.py"],
    "forensics": ["binwalk -e", "exiftool", "strings", "volatility -f mem.img"],
    "misc": ["file", "python3 solve.py", "nc host port", "xxd"],
}

OUTPUT_LINES = {
    "web": ["HTTP/1.1 200 OK", "Set-Cookie: session={h}", "GET /api/v2/item?id={h} -> 200",
            "<!-- debug: token={h} -->", "Found endpoint /admin/{h}", "X-Powered-By: PHP/7.4"],
    "pwn": ["RELRO: Partial RELRO", "0x{h}: mov rax, [rbp-0x{h}]", "Canary : found",
            "gdb$ x/20gx $rsp -> 0x{h}", "[*] Switching to interactive mode"],
    "rev": ["0x{h} <main>: call sub_{h}", "ASCII string: flag{{maybe_{h}}}", "cmp eax, 0x{h}",
            "Function sub_{h} referenced 3 times", "UPX!"],
    "crypto": ["n = {h}...", "e = 65537", "Possible factor found: {h}", "XOR key candidate: {h}",
               "Ciphertext block {h} decoded"],
    "forensics": ["Signature found at offset {h}: Zip archive", "PNG IHDR chunk at {h}",
                  "Deleted file recovered: secret_{h}.txt", "EXIF GPSLatitude: {h}"],
    "misc": ["file: data", "Connected to host on port {h}", "b'\\x{h}\\x00\\x01'"],
}


def rand_hex(n=6):
    return "".join(random.choices(string.hexdigits.lower(), k=n))


def fake_tool_output(category):
    pool = OUTPUT_LINES.get(category, OUTPUT_LINES["misc"])
    n_lines = random.randint(20, 70)
    lines = [random.choice(pool).format(h=rand_hex(random.choice([4, 6, 8, 12]))) for _ in range(n_lines)]
    return "\n".join(lines)


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def api_status():
    try:
        counts = {name: db[name].count_documents({}) for name in
                  ["semantic_memory", "runs", "episodic_log", "policy"]}
        connected = True
    except Exception:
        counts = {}
        connected = False
    return jsonify({
        "connected": connected,
        "db": "pentest_memory",
        "collections": counts,
        "ai_gateway_configured": ai_gateway is not None,
    })


@app.get("/api/challenges")
def api_challenges():
    return jsonify(CHALLENGES)


@app.get("/api/memory")
def api_memory():
    docs = list(semantic_col.find({}, {"_id": 0}).sort([("utility_score", -1), ("doc_id", 1)]))
    policy = get_policy(policy_col, POLICY_DEMO)
    return jsonify({"semantic_memory": docs, "policy": policy})


@app.get("/api/runs")
def api_runs():
    limit = int(request.args.get("limit", 25))
    only_sim = request.args.get("simulated", "1") != "0"
    q = {"simulated": True} if only_sim else {}
    docs = list(runs_col.find(q, {"_id": 0}).sort("timestamp", -1).limit(limit))
    return jsonify(docs)


@app.post("/api/reset")
def api_reset():
    """Return the demo to a clean slate: scores back to the prior, simulated
    runs/logs removed (real agent runs are kept), demo policy back to defaults.
    The real agent's policy (_id "global") is never touched here."""
    mem = semantic_col.update_many({}, {"$set": {
        "times_retrieved": 0, "times_led_to_success": 0, "utility_score": 0.5,
    }})
    runs = runs_col.delete_many({"simulated": True})
    logs = episodic_col.delete_many({"simulated": True})
    policy_col.replace_one({"_id": POLICY_DEMO}, default_policy(POLICY_DEMO), upsert=True)
    return jsonify({
        "memory_docs_reset": mem.modified_count,
        "runs_deleted": runs.deleted_count,
        "episodic_logs_deleted": logs.deleted_count,
        "policy_reset": True,
    })


@app.get("/api/simulate/stream")
def api_simulate_stream():
    challenge = CHALLENGES_BY_ID.get(request.args.get("challenge", ""))
    if challenge is None:
        return jsonify({"error": "unknown challenge id"}), 404

    def generate():
        policy = get_policy(policy_col, POLICY_DEMO)
        rag_top_k = policy.get("rag_top_k", 3)
        rag_inject_every = policy.get("rag_inject_every", 3)
        truncate_chars = policy.get("truncate_content_chars", 25000)
        alpha = policy.get("alpha", 0.6)
        beta = policy.get("beta", 0.4)

        mongo_rag = MongoRAG(semantic_col, top_k=rag_top_k, alpha=alpha, beta=beta,
                              inject_every=rag_inject_every)

        episode_id = f"sim-{challenge['id']}-{int(time.time())}"
        yield sse("start", {
            "episode_id": episode_id,
            "challenge": challenge,
            "policy": {
                "rag_top_k": rag_top_k, "rag_inject_every": rag_inject_every,
                "truncate_content_chars": truncate_chars, "alpha": alpha, "beta": beta,
            },
        })

        n_rounds = random.randint(9, 14)
        tools = TOOL_POOL.get(challenge["category"], TOOL_POOL["misc"])

        naive_total = BASE_OVERHEAD_TOKENS
        trunc_tokens_history = []
        memory_totals = []
        naive_totals = []
        used_hint_doc_ids = []
        best_utility_seen = 0.0

        for round_num in range(1, n_rounds + 1):
            hints = []
            if mongo_rag.should_retrieve(round_num):
                try:
                    hints = mongo_rag.get_hints(
                        challenge_name=challenge["name"],
                        challenge_category=challenge["category"],
                        challenge_description=challenge["description"],
                        current_round=round_num,
                    )
                except Exception as e:
                    yield sse("warning", {"round": round_num, "message": f"Atlas Vector Search failed: {e}"})
                    hints = []
                if hints:
                    used_hint_doc_ids.extend(h.doc["doc_id"] for h in hints)
                    best_utility_seen = max(best_utility_seen, max(h.utility for h in hints))
                    yield sse("hint", {"round": round_num, "hints": [h.to_dict() for h in hints]})

            tool_name = random.choice(tools)
            reasoning_source = "scripted"
            thought = f"Trying `{tool_name}` next based on findings so far in this {challenge['category']} challenge."
            if ai_gateway is not None:
                gw_text = ai_gateway.reason(
                    "You are a concise CTF assistant. In one short sentence, state the next step and why.",
                    f"Challenge: {challenge['name']} ({challenge['category']}). "
                    f"Round {round_num}/{n_rounds}. Chosen tool: {tool_name}.",
                )
                if gw_text:
                    thought = gw_text.strip()
                    reasoning_source = "gateway"

            full_output = fake_tool_output(challenge["category"])
            truncated_output = full_output[:truncate_chars]
            trunc_tokens = max(1, len(truncated_output) // 4)
            full_tokens = max(1, len(full_output) // 4)

            trunc_tokens_history.append(trunc_tokens)
            naive_total += full_tokens + 40

            hint_tokens = sum(len(h.format()) for h in hints) // 4 if hints else 0
            window_tokens = sum(trunc_tokens_history[-WINDOW:])
            memory_total = BASE_OVERHEAD_TOKENS + window_tokens + hint_tokens
            memory_totals.append(memory_total)
            naive_totals.append(naive_total)

            episodic_col.insert_one({
                "episode_id": episode_id,
                "round": round_num,
                "agent_role": "executor",
                "tool": tool_name,
                "full_output": full_output,
                "timestamp": time.time(),
                "simulated": True,
            })

            yield sse("round", {
                "round": round_num,
                "tool": tool_name,
                "thought": thought,
                "reasoning_source": reasoning_source,
                "output_preview": truncated_output[:350],
                "full_output_chars": len(full_output),
                "naive_context_tokens": naive_total,
                "memory_context_tokens": memory_total,
            })
            time.sleep(0.3)

        # Base 60% solve chance, +25% once a hint has proven itself (utility > 0.6),
        # so the "memory makes the agent better" trend shows quickly in the demo.
        solve_chance = 0.60 + (0.25 if best_utility_seen > 0.6 else 0.0)
        solved = random.random() < solve_chance
        tokens_used = sum(memory_totals)
        cost = round(tokens_used * COST_PER_TOKEN, 4)
        used_doc_ids = list(set(used_hint_doc_ids))

        runs_col.insert_one({
            "episode_id": episode_id,
            "category": challenge["category"],
            "challenge": challenge["name"],
            "solved": solved,
            "tokens_used": tokens_used,
            "cost": cost,
            "used_doc_ids": used_doc_ids,
            "timestamp": time.time(),
            "simulated": True,
        })

        policy_change = None
        try:
            policy_change = update_policy(policy_col, runs_col, policy_id=POLICY_DEMO)
        except Exception:
            policy_change = None

        yield sse("done", {
            "episode_id": episode_id,
            "solved": solved,
            "tokens_used": tokens_used,
            "cost": cost,
            "used_doc_ids": used_doc_ids,
            # Compare like with like: cumulative tokens sent across all rounds,
            # and the context size of the final round, for both approaches.
            "naive_tokens_used": sum(naive_totals),
            "final_naive_context": naive_total,
            "final_memory_context": memory_totals[-1],
            "policy_change": policy_change,
        })

    return Response(generate(), mimetype="text/event-stream")


def docker_preflight():
    """Return why Docker isn't ready for a real run, or None if it is."""
    def ok(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    try:
        if not ok(["docker", "info"]):
            return "Docker daemon is not running. Start Docker and try again."
    except FileNotFoundError:
        return "docker CLI not found on PATH."
    if not ok(["docker", "image", "inspect", REAL_AGENT_IMAGE]):
        return (f"Agent image {REAL_AGENT_IMAGE} is not built. Run: "
                f"docker build -t {REAL_AGENT_IMAGE} docker/multiagent")
    if not ok(["docker", "network", "inspect", REAL_AGENT_NETWORK]):
        if not ok(["docker", "network", "create", REAL_AGENT_NETWORK]):
            return f"Could not create Docker network {REAL_AGENT_NETWORK}."
    return None


def ensure_challenge_files(chal_path):
    """The dataset is sparse-checked-out with only JSON files; pull this
    challenge's full directory (binaries, docker-compose.yml, ...) on demand."""
    pattern = f"{chal_path}/*"
    # protectNTFS=false: some unrelated dataset paths contain '?' (invalid on Windows)
    git = ["git", "-C", str(NYUCTF_ROOT), "-c", "core.protectNTFS=false"]
    listed = subprocess.run(git + ["sparse-checkout", "list"], capture_output=True, text=True).stdout.split()
    if pattern not in listed:
        subprocess.run(git + ["sparse-checkout", "add", pattern], capture_output=True, text=True)
    return (NYUCTF_ROOT / chal_path / "challenge.json").exists()


@app.get("/api/run/stream")
def api_run_stream():
    """Run the real agent (run_rag.py) on one challenge and stream its output."""
    challenge = CHALLENGES_BY_ID.get(request.args.get("challenge", ""))
    if challenge is None:
        return jsonify({"error": "unknown challenge id"}), 404
    try:
        max_cost = min(max(float(request.args.get("max_cost", 1.0)), 0.05), REAL_AGENT_MAX_COST_CAP)
    except ValueError:
        max_cost = 1.0

    def generate():
        if not real_run_lock.acquire(blocking=False):
            yield sse("failed", {"message": "Another real agent run is already in progress."})
            return
        proc = None
        try:
            if not keys.get("OPENAI"):
                yield sse("failed", {"message": "OPENAI= is missing in keys.cfg."})
                return
            yield sse("log", {"line": "Checking Docker..."})
            err = docker_preflight()
            if err:
                yield sse("failed", {"message": err})
                return
            yield sse("log", {"line": f"Fetching challenge files for {challenge['path']}..."})
            if not ensure_challenge_files(challenge["path"]):
                yield sse("failed", {"message": f"Could not check out {challenge['path']} from the dataset."})
                return

            started = time.time()
            cmd = [sys.executable, "-u", "run_rag.py",
                   "--challenge", challenge["id"], "--split", "development",
                   "--rag-mode", "mongo_rag", "--config", REAL_AGENT_CONFIG,
                   "--max-cost", str(max_cost), "--logdir", "trajectories/webapp"]
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "COLUMNS": "120"}
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
            yield sse("start", {"challenge": challenge, "max_cost": max_cost,
                                "command": " ".join(cmd[1:])})
            for line in proc.stdout:
                yield sse("log", {"line": line.rstrip("\n")})
            proc.wait()

            # run_rag.py inserts exactly one `runs` doc at the end of a mongo_rag episode
            run_doc = runs_col.find_one(
                {"timestamp": {"$gte": started}, "simulated": {"$ne": True}},
                {"_id": 0}, sort=[("timestamp", -1)],
            )
            yield sse("done", {"exit_code": proc.returncode, "run": run_doc})
        finally:
            # Viewer closed the stream: stop the agent instead of letting it keep spending
            if proc is not None and proc.poll() is None:
                proc.terminate()
            real_run_lock.release()

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    print(f"Loaded {len(CHALLENGES)} NYU CTF challenges for the demo.")
    print(f"AI Gateway configured: {ai_gateway is not None}")
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)
