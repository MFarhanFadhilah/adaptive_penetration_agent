"""
Graph-RAG Module for CTF Solving.

Builds a knowledge graph where nodes represent:
  - CTF categories (rev, web, for, cry, pwn)
  - Attack techniques (buffer_overflow, sql_injection, etc.)
  - Specific tools (pwntools, sqlmap, sage, etc.)
  - Exploit patterns (ret2libc, ECDLP, LFI, etc.)

Edges represent relationships:
  - CATEGORY --[USES]--> TECHNIQUE
  - TECHNIQUE --[REQUIRES]--> TOOL
  - TECHNIQUE --[LEADS_TO]--> EXPLOIT_PATTERN
  - EXPLOIT_PATTERN --[HAS_HINT]--> HINT_TEXT

At query time, Graph-RAG:
  1. Identifies entry nodes from the challenge description (keyword matching)
  2. Performs multi-hop graph traversal (BFS, depth ≤ 3)
  3. Collects hint texts from reached HINT nodes
  4. Scores and deduplicates collected hints
  5. Returns formatted hints for injection

All graph traversal steps are logged for trajectory analysis.
"""
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ---------------------------------------------------------------------------
# Node / Edge definitions
# ---------------------------------------------------------------------------

class NodeType:
    CATEGORY = "category"
    TECHNIQUE = "technique"
    TOOL = "tool"
    PATTERN = "pattern"
    HINT = "hint"


@dataclass
class GraphHint:
    text: str
    source_path: List[str]   # traversal path that led to this hint
    relevance: float = 1.0

    def format(self) -> str:
        path_str = " → ".join(self.source_path)
        return (
            f"[Graph-RAG HINT | path: {path_str} | relevance={self.relevance:.2f}]\n"
            f"{self.text}"
        )

    def to_dict(self) -> dict:
        return {
            "hint_preview": self.text[:200],
            "path": self.source_path,
            "relevance": self.relevance,
        }


# ---------------------------------------------------------------------------
# Graph-RAG implementation
# ---------------------------------------------------------------------------

class GraphRAG:
    """
    Graph-RAG: multi-hop knowledge graph retrieval for CTF hints.

    Graph structure (built once in _build_graph):
      category → technique(s) → tool(s)
                               → pattern(s) → hint(s)

    Retrieval:
      1. Match challenge keywords to entry nodes.
      2. BFS traversal up to max_hops.
      3. Collect all HINT nodes reached.
      4. Score hints by path length (shorter = higher relevance).
    """

    def __init__(self, max_hops: int = 3, max_hints: int = 3):
        self.max_hops = max_hops
        self.max_hints = max_hints
        self.retrieval_log: List[dict] = []

        if HAS_NX:
            self.graph = nx.DiGraph()
        else:
            # Lightweight adjacency-list fallback
            self.graph = None
            self._adj: Dict[str, List[str]] = {}
            self._nodes: Dict[str, dict] = {}

        self._build_graph()
        self._build_keyword_index()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _add_node(self, node_id: str, node_type: str, **attrs):
        if HAS_NX:
            self.graph.add_node(node_id, node_type=node_type, **attrs)
        else:
            self._nodes[node_id] = {"node_type": node_type, **attrs}
            if node_id not in self._adj:
                self._adj[node_id] = []

    def _add_edge(self, src: str, dst: str, relation: str = "USES"):
        if HAS_NX:
            self.graph.add_edge(src, dst, relation=relation)
        else:
            self._adj.setdefault(src, []).append(dst)

    def _neighbors(self, node_id: str) -> List[str]:
        if HAS_NX:
            return list(self.graph.successors(node_id))
        return self._adj.get(node_id, [])

    def _node_attr(self, node_id: str, attr: str, default=None):
        if HAS_NX:
            return self.graph.nodes[node_id].get(attr, default)
        return self._nodes.get(node_id, {}).get(attr, default)

    def _build_graph(self):
        """Construct the full CTF knowledge graph."""

        # ---- CATEGORIES ----
        for cat in ["rev", "web", "for", "cry", "pwn"]:
            self._add_node(cat, NodeType.CATEGORY, label=cat.upper())

        # ============================================================
        # REV branch
        # ============================================================
        self._add_node("rev_static_analysis", NodeType.TECHNIQUE,
                        label="Static Binary Analysis",
                        keywords=["strings", "binary", "static", "elf", "pe"])
        self._add_node("rev_dynamic_analysis", NodeType.TECHNIQUE,
                        label="Dynamic Analysis / Debugging",
                        keywords=["gdb", "ltrace", "strace", "debug", "dynamic"])
        self._add_node("rev_decompile", NodeType.TECHNIQUE,
                        label="Decompilation",
                        keywords=["decompile", "ghidra", "elf", "binary"])
        self._add_edge("rev", "rev_static_analysis")
        self._add_edge("rev", "rev_dynamic_analysis")
        self._add_edge("rev", "rev_decompile")

        # Tools
        self._add_node("tool_strings", NodeType.TOOL, label="strings",
                        keywords=["strings", "hardcoded", "extracted"])
        self._add_node("tool_ghidra", NodeType.TOOL, label="ghidra/radare2",
                        keywords=["ghidra", "radare2", "r2", "decompile"])
        self._add_node("tool_gdb", NodeType.TOOL, label="gdb+pwndbg",
                        keywords=["gdb", "debug", "breakpoint"])
        self._add_node("tool_angr", NodeType.TOOL, label="angr (symbolic)",
                        keywords=["angr", "symbolic", "solver"])
        self._add_edge("rev_static_analysis", "tool_strings")
        self._add_edge("rev_decompile", "tool_ghidra")
        self._add_edge("rev_dynamic_analysis", "tool_gdb")
        self._add_edge("rev_dynamic_analysis", "tool_angr")

        # Patterns + Hints
        self._add_node("pat_xor_encode", NodeType.PATTERN,
                        label="XOR-Encoded Strings",
                        keywords=["xor", "encode", "decode", "key"])
        self._add_edge("rev_static_analysis", "pat_xor_encode")
        self._add_node("hint_rev_strings", NodeType.HINT,
                        text="Run 'strings ./binary | grep -i flag' and 'ltrace ./binary' to find hardcoded or plaintext flags. Check .rodata section: 'objdump -s -j .rodata binary'.")
        self._add_node("hint_rev_xor", NodeType.HINT,
                        text="For XOR-encoded flags: look for a loop with XOR in decompiled code. Try all 256 single-byte XOR keys: python3 -c \"data=bytes([...]); [print(bytes(b^k for b in data)) for k in range(256)]\"")
        self._add_node("hint_rev_gdb", NodeType.HINT,
                        text="In GDB: 'b main; run; info functions; disas <func>'. Use 'x/s $rdi' at strcmp/memcmp to see expected value. pwndbg's 'context' shows registers. Try 'finish' to reach function return.")
        self._add_edge("tool_strings", "hint_rev_strings")
        self._add_edge("pat_xor_encode", "hint_rev_xor")
        self._add_edge("tool_gdb", "hint_rev_gdb")

        # ============================================================
        # WEB branch
        # ============================================================
        self._add_node("web_sqli", NodeType.TECHNIQUE, label="SQL Injection",
                        keywords=["sql", "injection", "sqli", "database", "login"])
        self._add_node("web_lfi", NodeType.TECHNIQUE, label="LFI/Path Traversal",
                        keywords=["lfi", "path traversal", "include", "file"])
        self._add_node("web_ssti", NodeType.TECHNIQUE, label="SSTI/Template Injection",
                        keywords=["ssti", "template", "jinja2", "twig"])
        self._add_node("web_recon", NodeType.TECHNIQUE, label="Web Recon/Enumeration",
                        keywords=["recon", "enumerate", "dirbusting", "robots"])
        self._add_node("web_upload", NodeType.TECHNIQUE, label="File Upload Bypass",
                        keywords=["upload", "php", "file", "bypass", "extension"])
        self._add_edge("web", "web_sqli")
        self._add_edge("web", "web_lfi")
        self._add_edge("web", "web_ssti")
        self._add_edge("web", "web_recon")
        self._add_edge("web", "web_upload")

        self._add_node("tool_sqlmap", NodeType.TOOL, label="sqlmap",
                        keywords=["sqlmap", "sql injection", "automated"])
        self._add_node("tool_curl", NodeType.TOOL, label="curl/burp",
                        keywords=["curl", "http", "request", "burp"])
        self._add_node("tool_gobuster", NodeType.TOOL, label="gobuster",
                        keywords=["gobuster", "dirb", "fuzz", "directories"])
        self._add_edge("web_sqli", "tool_sqlmap")
        self._add_edge("web_recon", "tool_curl")
        self._add_edge("web_recon", "tool_gobuster")
        self._add_edge("web_lfi", "tool_curl")

        self._add_node("hint_sqli_login", NodeType.HINT,
                        text="SQL injection login bypass: try username=admin'-- password=anything. Test with: curl -d 'user=admin%27--&pass=x' URL. Use sqlmap: 'sqlmap -u URL --data \"user=x&pass=y\" --dbs --dump'.")
        self._add_node("hint_lfi_php", NodeType.HINT,
                        text="PHP LFI: test ?page=../../../etc/passwd. For PHP source: ?page=php://filter/convert.base64-encode/resource=index. For RCE via log poisoning: inject PHP in User-Agent then include /var/log/apache2/access.log.")
        self._add_node("hint_ssti_jinja", NodeType.HINT,
                        text="Jinja2 SSTI: test {{7*7}}=49. RCE: {{config.__class__.__init__.__globals__['os'].popen('cat /flag').read()}}. Or: {{''.__class__.__mro__[1].__subclasses__()[<id>]('cat /flag',shell=True,stdout=-1).communicate()[0].strip()}}")
        self._add_node("hint_web_recon", NodeType.HINT,
                        text="Web recon checklist: (1) curl -I URL for headers, (2) curl URL/robots.txt, (3) curl URL/.git/HEAD for git exposure, (4) Check page source for comments/JS includes, (5) gobuster dir -u URL -w /usr/share/wordlists/dirb/common.txt")
        self._add_edge("web_sqli", "hint_sqli_login")
        self._add_edge("web_lfi", "hint_lfi_php")
        self._add_edge("web_ssti", "hint_ssti_jinja")
        self._add_edge("web_recon", "hint_web_recon")

        # ============================================================
        # FOR branch
        # ============================================================
        self._add_node("for_stego", NodeType.TECHNIQUE, label="Steganography",
                        keywords=["stego", "steganography", "hidden", "image", "lsb"])
        self._add_node("for_qr", NodeType.TECHNIQUE, label="QR Code / Barcode Analysis",
                        keywords=["qr", "barcode", "1black0white", "binary image"])
        self._add_node("for_network", NodeType.TECHNIQUE, label="Network Forensics",
                        keywords=["pcap", "wireshark", "network", "capture"])
        self._add_node("for_file_carve", NodeType.TECHNIQUE, label="File Carving",
                        keywords=["binwalk", "carve", "extract", "embedded"])
        self._add_edge("for", "for_stego")
        self._add_edge("for", "for_qr")
        self._add_edge("for", "for_network")
        self._add_edge("for", "for_file_carve")

        self._add_node("tool_binwalk", NodeType.TOOL, label="binwalk",
                        keywords=["binwalk", "extract", "embedded"])
        self._add_node("tool_steghide", NodeType.TOOL, label="steghide/zsteg",
                        keywords=["steghide", "zsteg", "stegsolve"])
        self._add_node("tool_wireshark", NodeType.TOOL, label="wireshark/tshark",
                        keywords=["wireshark", "tshark", "pcap", "network"])
        self._add_edge("for_stego", "tool_steghide")
        self._add_edge("for_qr", "tool_binwalk")
        self._add_edge("for_network", "tool_wireshark")
        self._add_edge("for_file_carve", "tool_binwalk")

        self._add_node("hint_qr_reconstruct", NodeType.HINT,
                        text="For QR from numeric data (1black0white): each decimal = binary row. Python3: from PIL import Image; rows=[bin(int(n))[2:].zfill(31) for n in open('qr_code.txt').read().split()]; img=Image.new('1',(len(rows[0]),len(rows))); [img.putpixel((j,i),int(rows[i][j])) for i in range(len(rows)) for j in range(len(rows[0]))]; img.save('qr.png'). Then: zbarimg qr.png or use PIL.ImageOps.")
        self._add_node("hint_stego_lsb", NodeType.HINT,
                        text="LSB steganography: use 'zsteg image.png' to auto-detect. Manual: python3 -c \"from PIL import Image; img=Image.open('f.png'); data=''.join([str(img.getpixel((x,y))[0]&1) for y in range(img.height) for x in range(img.width)])\". steghide extract -sf img.jpg (password may be empty).")
        self._add_node("hint_pcap_http", NodeType.HINT,
                        text="Wireshark tips: File>Export Objects>HTTP for files. Follow TCP stream: right-click>Follow>TCP Stream. tshark: 'tshark -r f.pcap -q -z http,tree' or '-T fields -e http.file_data'. For credentials look in POST data or Basic Auth headers.")
        self._add_edge("for_qr", "hint_qr_reconstruct")
        self._add_edge("for_stego", "hint_stego_lsb")
        self._add_edge("for_network", "hint_pcap_http")

        # ============================================================
        # CRY branch
        # ============================================================
        self._add_node("cry_rsa", NodeType.TECHNIQUE, label="RSA Cryptanalysis",
                        keywords=["rsa", "modulus", "factor", "e", "d", "prime"])
        self._add_node("cry_ecc", NodeType.TECHNIQUE, label="ECC / Elliptic Curve",
                        keywords=["ecc", "elliptic", "curve", "ecdsa", "ecdlp", "super_curve"])
        self._add_node("cry_classical", NodeType.TECHNIQUE, label="Classical Ciphers",
                        keywords=["vigenere", "caesar", "rot", "classical", "substitution"])
        self._add_node("cry_aes", NodeType.TECHNIQUE, label="AES/Symmetric Attacks",
                        keywords=["aes", "cbc", "padding oracle", "symmetric"])
        self._add_edge("cry", "cry_rsa")
        self._add_edge("cry", "cry_ecc")
        self._add_edge("cry", "cry_classical")
        self._add_edge("cry", "cry_aes")

        self._add_node("tool_sage", NodeType.TOOL, label="SageMath",
                        keywords=["sage", "sagemath", "math", "group", "ecc"])
        self._add_node("tool_openssl", NodeType.TOOL, label="openssl/rsatool",
                        keywords=["openssl", "rsa", "asn1", "pem", "der"])
        self._add_edge("cry_ecc", "tool_sage")
        self._add_edge("cry_rsa", "tool_sage")
        self._add_edge("cry_rsa", "tool_openssl")

        self._add_node("pat_pohlig_hellman", NodeType.PATTERN,
                        label="Pohlig-Hellman ECDLP",
                        keywords=["pohlig", "smooth order", "subgroup", "crt"])
        self._add_node("pat_smart_attack", NodeType.PATTERN,
                        label="Smart's Attack (anomalous curve)",
                        keywords=["smart", "anomalous", "p-adic", "lift"])
        self._add_node("pat_ecdsa_nonce", NodeType.PATTERN,
                        label="ECDSA Nonce Reuse",
                        keywords=["nonce", "reuse", "ecdsa", "k reuse", "two signatures"])
        self._add_edge("cry_ecc", "pat_pohlig_hellman")
        self._add_edge("cry_ecc", "pat_smart_attack")
        self._add_edge("cry_ecc", "pat_ecdsa_nonce")

        self._add_node("hint_ecc_pohlig", NodeType.HINT,
                        text="Pohlig-Hellman for ECDLP: E=EllipticCurve(GF(p),[a,b]); n=E.order(); f=factor(n). If n has small primes, use: from sage.groups.generic import discrete_log; k=discrete_log(Q, G, ord=n, operation='+'). CRT if subgroup-based. super_curve likely has smooth order!")
        self._add_node("hint_ecc_smart", NodeType.HINT,
                        text="Smart's attack: if #E(Fp)==p (anomalous), the ECDLP lifts to Z/pZ via p-adic logarithm. Sage snippet: E_lift=EllipticCurve(Qp(p,2),[a,b]); use lift and formal logarithm. Many CTF writeups available for this.")
        self._add_node("hint_ecdsa_nonce", NodeType.HINT,
                        text="ECDSA nonce reuse: given (r1,s1,msg1) and (r1,s2,msg2) with same r → same k. k=(hash(m1)-hash(m2))*modular_inverse(s1-s2, n) % n. Then private_key=(s1*k - hash(m1))*modular_inverse(r1,n) % n.")
        self._add_edge("pat_pohlig_hellman", "hint_ecc_pohlig")
        self._add_edge("pat_smart_attack", "hint_ecc_smart")
        self._add_edge("pat_ecdsa_nonce", "hint_ecdsa_nonce")

        # ============================================================
        # PWN branch
        # ============================================================
        self._add_node("pwn_bof", NodeType.TECHNIQUE, label="Buffer Overflow",
                        keywords=["buffer overflow", "bof", "overflow", "stack", "smash"])
        self._add_node("pwn_rop", NodeType.TECHNIQUE, label="Return Oriented Programming",
                        keywords=["rop", "ret2libc", "gadget", "return", "chain"])
        self._add_node("pwn_fmt", NodeType.TECHNIQUE, label="Format String",
                        keywords=["format string", "printf", "%x", "%n", "leak"])
        self._add_node("pwn_heap", NodeType.TECHNIQUE, label="Heap Exploitation",
                        keywords=["heap", "malloc", "free", "use after free", "double free"])
        self._add_edge("pwn", "pwn_bof")
        self._add_edge("pwn", "pwn_rop")
        self._add_edge("pwn", "pwn_fmt")
        self._add_edge("pwn", "pwn_heap")
        self._add_edge("pwn_bof", "pwn_rop")

        self._add_node("tool_pwntools", NodeType.TOOL, label="pwntools",
                        keywords=["pwntools", "pwn", "exploit", "p64", "cyclic"])
        self._add_node("tool_ropgadget", NodeType.TOOL, label="ROPgadget/ropper",
                        keywords=["ropgadget", "ropper", "gadget", "rop chain"])
        self._add_node("tool_checksec", NodeType.TOOL, label="checksec",
                        keywords=["checksec", "nx", "aslr", "pie", "canary", "protections"])
        self._add_edge("pwn_bof", "tool_pwntools")
        self._add_edge("pwn_rop", "tool_ropgadget")
        self._add_edge("pwn_bof", "tool_checksec")

        self._add_node("pat_ret2win", NodeType.PATTERN, label="ret2win (direct win())",
                        keywords=["win", "ret2win", "function", "simple overflow"])
        self._add_node("pat_ret2libc", NodeType.PATTERN, label="ret2libc",
                        keywords=["ret2libc", "system", "/bin/sh", "got", "plt"])
        self._add_edge("pwn_rop", "pat_ret2win")
        self._add_edge("pwn_rop", "pat_ret2libc")

        self._add_node("hint_pwn_basic", NodeType.HINT,
                        text="Basic BOF workflow: checksec ./binary → python3 -c 'from pwn import *; print(cyclic(200).decode())' to find offset → gdb: run < <(python3 -c 'from pwn import *; print(cyclic(200).decode())') → info registers rsp/eip at crash → cyclic_find(0x<crash_val>).")
        self._add_node("hint_pwn_ret2win", NodeType.HINT,
                        text="ret2win: nm ./binary | grep win → find win() address → from pwn import *; p=process('./binary'); offset=<N>; payload=b'A'*offset+p64(win_addr); p.sendline(payload); p.interactive(). For bigboy: check if there's a variable that needs a specific value (like 0xdeadbeef).")
        self._add_node("hint_pwn_ret2libc", NodeType.HINT,
                        text="ret2libc with ASLR: leak libc via puts(GOT[puts]) → calculate libc_base = leak - puts_offset → system = libc_base + system_offset → /bin/sh = libc_base + binsh_offset → overflow → ret;pop_rdi;/bin/sh_addr;system_addr.")
        self._add_edge("pwn_bof", "hint_pwn_basic")
        self._add_edge("pat_ret2win", "hint_pwn_ret2win")
        self._add_edge("pat_ret2libc", "hint_pwn_ret2libc")

    def _build_keyword_index(self):
        """Map keywords to node IDs for entry-point lookup."""
        self._kw_to_nodes: Dict[str, List[str]] = {}

        def index_node(node_id: str):
            kws = self._node_attr(node_id, "keywords") or []
            for kw in kws:
                for token in re.findall(r"[a-z0-9_]+", kw.lower()):
                    self._kw_to_nodes.setdefault(token, []).append(node_id)

        if HAS_NX:
            for node_id in self.graph.nodes:
                index_node(node_id)
        else:
            for node_id in self._nodes:
                index_node(node_id)

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------

    def _match_entry_nodes(self, query: str) -> List[Tuple[str, int]]:
        """
        Return (node_id, match_count) sorted by match_count descending.
        Category nodes are always included if the category appears in query.
        """
        tokens = re.findall(r"[a-z0-9_]+", query.lower())
        counts: Dict[str, int] = {}
        for token in tokens:
            for nid in self._kw_to_nodes.get(token, []):
                counts[nid] = counts.get(nid, 0) + 1
        # Always include the category node if it matches
        for cat in ["rev", "web", "for", "cry", "pwn"]:
            if cat in tokens:
                counts[cat] = counts.get(cat, 0) + 3  # boost category node
        return sorted(counts.items(), key=lambda x: -x[1])

    def _bfs_collect_hints(
        self, entry_nodes: List[str]
    ) -> List[Tuple[GraphHint, int]]:
        """BFS from entry nodes; collect all HINT nodes with their depth."""
        visited: Set[str] = set()
        queue: deque = deque()

        for nid, _ in entry_nodes[:5]:  # top-5 entry points
            queue.append((nid, 0, [nid]))
            visited.add(nid)

        collected: List[Tuple[GraphHint, int]] = []

        while queue:
            node_id, depth, path = queue.popleft()
            if depth > self.max_hops:
                continue

            ntype = self._node_attr(node_id, "node_type")
            if ntype == NodeType.HINT:
                hint_text = self._node_attr(node_id, "text", "")
                # Relevance decreases with depth
                relevance = max(0.1, 1.0 - depth * 0.25)
                collected.append(
                    (GraphHint(text=hint_text, source_path=path, relevance=relevance), depth)
                )

            for neighbor in self._neighbors(node_id):
                if neighbor not in visited and depth + 1 <= self.max_hops:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1, path + [neighbor]))

        return collected

    def get_hints(
        self,
        challenge_name: str,
        challenge_category: str,
        challenge_description: str,
        current_plan: str = "",
        current_round: int = 1,
    ) -> List[GraphHint]:
        """Graph-RAG retrieval pipeline."""
        query = (
            f"{challenge_category} {challenge_name} {challenge_description} "
            f"{current_plan[:300]}"
        )

        log_entry = {
            "round": current_round,
            "query_preview": query[:300],
            "entry_nodes": [],
            "hints_collected": 0,
            "injected_hints": [],
            "timestamp": time.time(),
        }

        entry_nodes = self._match_entry_nodes(query)
        log_entry["entry_nodes"] = [
            {"node_id": nid, "match_count": cnt} for nid, cnt in entry_nodes[:8]
        ]

        hint_candidates = self._bfs_collect_hints(entry_nodes)
        # Sort by relevance (depth-based)
        hint_candidates.sort(key=lambda x: -x[0].relevance)
        # Deduplicate by text
        seen_texts: Set[str] = set()
        unique_hints: List[GraphHint] = []
        for hint, _ in hint_candidates:
            key = hint.text[:80]
            if key not in seen_texts:
                seen_texts.add(key)
                unique_hints.append(hint)

        result = unique_hints[: self.max_hints]
        log_entry["hints_collected"] = len(hint_candidates)
        log_entry["injected_hints"] = [h.to_dict() for h in result]
        self.retrieval_log.append(log_entry)
        return result

    def format_hints_for_injection(self, hints: List[GraphHint]) -> str:
        """Format graph hints for injection into planner context."""
        if not hints:
            return ""
        parts = ["=== GRAPH-RAG KNOWLEDGE HINTS (multi-hop graph traversal) ==="]
        for i, h in enumerate(hints, 1):
            path_str = " → ".join(h.source_path)
            parts.append(f"\n[Graph-Hint {i} | path: {path_str} | relevance={h.relevance:.2f}]")
            parts.append(h.text)
        parts.append("=== END GRAPH HINTS ===")
        return "\n".join(parts)

    def dump_log(self) -> List[dict]:
        return self.retrieval_log

    def describe_graph(self) -> str:
        """Return a human-readable summary of the graph structure."""
        if HAS_NX:
            n_nodes = self.graph.number_of_nodes()
            n_edges = self.graph.number_of_edges()
        else:
            n_nodes = len(self._nodes)
            n_edges = sum(len(v) for v in self._adj.values())
        return (
            f"Graph-RAG knowledge graph: {n_nodes} nodes, {n_edges} edges. "
            f"Node types: category, technique, tool, pattern, hint. "
            f"Max traversal hops: {self.max_hops}."
        )
