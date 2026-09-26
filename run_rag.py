"""
run_rag.py - Assignment 3: RAG-Augmented CTF Agent

Supports three modes:
  --rag-mode self_rag   : Uses Self-RAG (BM25 retrieval + LLM self-critique)
  --rag-mode graph_rag  : Uses Graph-RAG (knowledge graph multi-hop traversal)
  --rag-mode combined   : Uses both Self-RAG and Graph-RAG together
  --rag-mode mongo_rag  : Uses MongoDB Atlas Vector Search; RAG settings are read
                          from the self-tuning `policy` document in Atlas

Example:
  python run_rag.py --challenge 2023q-rev-baby_s_third --split test \\
      --config configs/rag/self_rag_config.yaml \\
      --rag-mode self_rag --logdir trajectories/self_rag -n rag_run
"""

import argparse
import sys
from pathlib import Path

import yaml

from nyuctf.dataset import CTFDataset
from nyuctf.challenge import CTFChallenge

from nyuctf_multiagent.environment import CTFEnvironment
from nyuctf_multiagent.backends import MODELS, Role
from nyuctf_multiagent.prompting import PromptManager
from nyuctf_multiagent.agent import PlannerAgent, ExecutorAgent, AutoPromptAgent
from nyuctf_multiagent.logging import logger
from nyuctf_multiagent.utils import APIKeys, load_common_options, get_log_filename, load_config
from nyuctf_multiagent.config import Config
from nyuctf_multiagent.rag_agent import RAGPlannerExecutorSystem
from nyuctf_multiagent.rag import CTFKnowledgeBase, SelfRAG, GraphRAG


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="RAG-Augmented Multi-Agent CTF Solver (Assignment 3)"
)
load_common_options(parser)
parser.add_argument("--logdir", default="trajectories/rag", type=str, help="Log directory")
parser.add_argument(
    "--config", default=None,
    help="YAML config file. If not provided, defaults to configs/rag/self_rag_config.yaml"
)
parser.add_argument(
    "--rag-mode", default=None,
    choices=["self_rag", "graph_rag", "mongo_rag", "combined"],
    help="RAG mode: self_rag, graph_rag, mongo_rag, or combined (default: config's rag.mode, else self_rag)"
)
parser.add_argument("--planner-model", default=None)
parser.add_argument("--executor-model", default=None)
parser.add_argument("--autoprompter-model", default=None)
parser.add_argument("--max-cost", default=0.0, type=float)
parser.add_argument("--enable-autoprompt", action="store_true")
parser.add_argument("--strict", action="store_true")
# RAG settings default to None so an explicit CLI value can be told apart from
# "not given". Precedence: CLI > Mongo policy (mongo_rag mode) > YAML > default.
parser.add_argument("--rag-relevance-threshold", default=None, type=float,
                    help="Self-RAG relevance threshold (0-10), default 5.0")
parser.add_argument("--rag-top-k", default=None, type=int,
                    help="Number of candidate documents for RAG retrieval, default 3")
parser.add_argument("--rag-max-hops", default=None, type=int,
                    help="Graph-RAG maximum traversal hops, default 3")
parser.add_argument("--rag-inject-every", default=None, type=int,
                    help="Inject RAG hints every N planner rounds, default 3")

args = parser.parse_args()


def resolve_setting(cli_value, policy_value, yaml_value, default):
    """First value that was actually set wins: CLI > Mongo policy > YAML > default."""
    for value in (cli_value, policy_value, yaml_value):
        if value is not None:
            return value
    return default


logger.set(quiet=args.quiet, debug=args.debug)

# ---------------------------------------------------------------------------
# Dataset / challenge setup
# ---------------------------------------------------------------------------
if args.dataset is not None:
    dataset = CTFDataset(dataset_json=args.dataset)
else:
    dataset = CTFDataset(split=args.split)

challenge = CTFChallenge(dataset.get(args.challenge), dataset.basedir)
logfile = get_log_filename(args, challenge)

logger.print(f"Logging to {str(logfile)}", force=True)
if logfile.exists() and args.skip_existing:
    logger.print("Skipping as log file exists", force=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
if args.config:
    config_f = Path(args.config)
else:
    config_d = Path("configs/rag")
    config_map = {
        "self_rag": "self_rag_config.yaml",
        "graph_rag": "graph_rag_config.yaml",
        "combined": "combined_rag_config.yaml",
        "mongo_rag": "self_rag_config.yaml",  # same models/prompts; RAG settings come from the Mongo policy
    }
    config_f = config_d / config_map[args.rag_mode or "self_rag"]

logger.print(f"Using config: {str(config_f)}", force=True)
config = load_config(config_f, args=args)

if args.strict:
    config.planner.strict = True
    config.executor.strict = True
    config.autoprompter.strict = True

rag_cfg = {}
try:
    raw_yaml = yaml.safe_load(config_f.open("r"))
    rag_cfg = raw_yaml.get("rag", {}) or {}
except Exception:
    pass

keys = APIKeys(args.keys)

rag_mode = resolve_setting(args.rag_mode, None, rag_cfg.get("mode"), "self_rag")

# In mongo_rag mode the harness settings live in the `policy` document in Atlas
# (see mongo/policy.py), which update_policy() rewrites from measured outcomes.
policy = {}
mongo_db = None
if rag_mode == "mongo_rag":
    from pymongo import MongoClient
    from mongo.policy import get_policy

    mongo_db = MongoClient(keys["MONGODB_URI"])["pentest_memory"]
    policy = get_policy(mongo_db["policy"])

relevance_threshold = resolve_setting(args.rag_relevance_threshold, None, rag_cfg.get("relevance_threshold"), 5.0)
top_k = resolve_setting(args.rag_top_k, policy.get("rag_top_k"), rag_cfg.get("top_k"), 3)
max_hints_per_round = rag_cfg.get("max_hints_per_round", 2)
max_hops = resolve_setting(args.rag_max_hops, None, rag_cfg.get("max_hops"), 3)
max_graph_hints = rag_cfg.get("max_hints", 3)
inject_every = resolve_setting(args.rag_inject_every, policy.get("rag_inject_every"), rag_cfg.get("inject_every"), 3)
alpha = policy.get("alpha", 0.6)
beta = policy.get("beta", 0.4)
truncate_content = policy.get("truncate_content_chars")  # None -> keep Conversation's default

logger.print(f"[RAG Mode: {rag_mode.upper()}]", force=True)
if policy:
    logger.print(
        f"[Policy] v{policy['version']} loaded from Atlas: top_k={top_k}, inject_every={inject_every}, "
        f"truncate={truncate_content}, alpha={alpha}, beta={beta}",
        force=True
    )

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
environment = CTFEnvironment(challenge, args.container_image, args.container_network)

# ---------------------------------------------------------------------------
# RAG modules
# ---------------------------------------------------------------------------
knowledge_base = CTFKnowledgeBase()
logger.print(f"Loaded CTF knowledge base: {len(knowledge_base.documents)} documents", force=True)

self_rag = None
graph_rag = None

if rag_mode in ("self_rag", "combined"):
    self_rag = SelfRAG(
        knowledge_base=knowledge_base,
        api_key=keys["GEMINI"],
        model=config.planner.model,
        relevance_threshold=relevance_threshold,
        top_k=top_k,
        max_hints_per_round=max_hints_per_round,
    )
    logger.print(
        f"[Self-RAG] Initialized: threshold={relevance_threshold}, top_k={top_k}",
        force=True
    )

if rag_mode in ("graph_rag", "combined"):
    graph_rag = GraphRAG(max_hops=max_hops, max_hints=max_graph_hints)
    logger.print(
        f"[Graph-RAG] Initialized: {graph_rag.describe_graph()}",
        force=True
    )

mongo_rag = None
runs_collection = None
policy_collection = None

if rag_mode in ("mongo_rag",):
    from mongo.mongo_rag import MongoRAG, INDEX_NAME as MONGO_RAG_INDEX_NAME

    mongo_rag = MongoRAG(
        collection=mongo_db["semantic_memory"],
        top_k=top_k,
        alpha=alpha,
        beta=beta,
        max_hints_per_round=max_hints_per_round,
        inject_every=inject_every,
    )
    runs_collection = mongo_db["runs"]
    policy_collection = mongo_db["policy"]
    logger.print(
        f"[Mongo-RAG] Initialized: top_k={top_k}, index={MONGO_RAG_INDEX_NAME}",
        force=True
    )

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
autoprompter_backend_cls = MODELS[config.autoprompter.model]
autoprompter_backend = autoprompter_backend_cls(
    Role.AUTOPROMPTER, config.autoprompter.model,
    environment.get_toolset(config.autoprompter.toolset),
    keys[autoprompter_backend_cls.NAME.upper()], config
)
autoprompter_prompter = PromptManager(
    config_f.parent / config.autoprompter.prompt, challenge, environment
)
autoprompter = AutoPromptAgent(
    environment, challenge, autoprompter_prompter,
    autoprompter_backend, max_rounds=config.autoprompter.max_rounds
)
if config.experiment.enable_autoprompt or args.enable_autoprompt:
    autoprompter.enable_autoprompt()

planner_backend_cls = MODELS[config.planner.model]
planner_backend = planner_backend_cls(
    Role.PLANNER, config.planner.model,
    environment.get_toolset(config.planner.toolset),
    keys[planner_backend_cls.NAME.upper()], config
)
planner_prompter = PromptManager(
    config_f.parent / config.planner.prompt, challenge, environment
)
planner = PlannerAgent(
    environment, challenge, planner_prompter,
    planner_backend, max_rounds=config.planner.max_rounds
)

executor_backend_cls = MODELS[config.executor.model]
executor_backend = executor_backend_cls(
    Role.EXECUTOR, config.executor.model,
    environment.get_toolset(config.executor.toolset),
    keys[executor_backend_cls.NAME.upper()], config
)
executor_prompter = PromptManager(
    config_f.parent / config.executor.prompt, challenge, environment
)
executor = ExecutorAgent(
    environment, challenge, executor_prompter,
    executor_backend, max_rounds=config.executor.max_rounds
)
executor.conversation.len_observations = config.executor.len_observations

executor_b = None
if config.executor_b is not None:
    executor_b_backend_cls = MODELS[config.executor_b.model]
    executor_b_backend = executor_b_backend_cls(
        Role.EXECUTOR, config.executor_b.model,
        environment.get_toolset(config.executor_b.toolset),
        keys[executor_b_backend_cls.NAME.upper()], config
    )
    executor_b_prompter = PromptManager(
        config_f.parent / config.executor_b.prompt, challenge, environment
    )
    executor_b = ExecutorAgent(
        environment, challenge, executor_b_prompter,
        executor_b_backend, max_rounds=config.executor_b.max_rounds
    )
    executor_b.conversation.len_observations = config.executor_b.len_observations

if truncate_content is not None:
    # Executors are re-created per delegated task via ExecutorAgent.new(), which carries this over
    for agent in (planner, executor, executor_b):
        if agent is not None:
            agent.conversation.truncate_content = truncate_content

# ---------------------------------------------------------------------------
# Run RAG-augmented system
# ---------------------------------------------------------------------------
with RAGPlannerExecutorSystem(
    environment, challenge, autoprompter, planner, executor,
    max_cost=config.experiment.max_cost,
    logfile=logfile,
    executor_b=executor_b,
    self_rag=self_rag,
    graph_rag=graph_rag,
    mongo_rag=mongo_rag,
    runs_collection=runs_collection,
    policy_collection=policy_collection,
    policy_version=policy.get("version"),
    episodic_collection=mongo_db["episodic_log"] if rag_mode == "mongo_rag" else None,
    rag_inject_every=inject_every,
    mode=rag_mode,
) as system:
    system.run()
