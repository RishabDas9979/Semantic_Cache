"""Benchmark: threshold sweep on paraphrase vs. unrelated queries, plus latency.

Measures hit rate on paraphrases (should hit), false-positive rate on
different-intent questions (should miss), and per-lookup latency.
Uses a simulated LLM latency so cost savings are explicit.
"""
import statistics
import time

from semcache import SemanticCache

# (canonical question, [paraphrases that should HIT])
GROUPS = [
    ("How do I reset my password?", ["How can I reset my password", "I forgot my password, how do I reset it?", "Steps to reset password"]),
    ("What is your refund policy?", ["Tell me about the refund policy", "How do refunds work?", "what's the policy on refunds"]),
    ("How do I cancel my subscription?", ["Cancel my subscription please", "How can I cancel a subscription?", "steps to cancel subscription"]),
    ("Where can I download my invoice?", ["How do I download an invoice?", "Where do I find my invoices to download", "download invoice where"]),
    ("How do I change my email address?", ["Change the email on my account", "How can I update my email address?", "steps to change email address"]),
    ("What payment methods do you accept?", ["Which payment methods are accepted?", "What kinds of payment do you accept", "accepted payment methods?"]),
    ("How do I enable two factor authentication?", ["Enable two-factor authentication how?", "How can I turn on 2 factor authentication", "steps to enable two factor authentication"]),
    ("How long does shipping take?", ["What is the shipping time?", "How long will shipping take", "how many days does shipping take"]),
]
# Different-intent queries that share vocabulary and should MISS.
NEGATIVES = [
    "How do I reset my router?",
    "What is your privacy policy?",
    "How do I upgrade my subscription?",
    "Where can I upload my logo?",
    "How do I change my password?",
    "What currencies do you support?",
    "How do I disable two factor authentication?",
    "How long does a refund take?",
]
LLM_LATENCY = 0.8  # seconds, simulated


def run(threshold: float):
    c = SemanticCache(threshold=threshold)
    for q, _ in GROUPS:
        c.store(q, f"answer::{q}")
    pos = [p for _, ps in GROUPS for p in ps]
    hits = sum(1 for p in pos if c.lookup(p))
    fps = sum(1 for n in NEGATIVES if c.lookup(n))
    return hits / len(pos), fps / len(NEGATIVES)


def latency(n_entries: int = 10_000, trials: int = 200):
    c = SemanticCache(threshold=0.8, max_entries=n_entries + 10)
    for i in range(n_entries):
        c.store(f"synthetic question number {i} about topic {i % 97}", "x")
    times = []
    for i in range(trials):
        t = time.perf_counter()
        c.lookup(f"synthetic question number {i} about topic {i % 97}")
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return statistics.mean(times), times[int(0.95 * len(times)) - 1]


if __name__ == "__main__":
    print("Threshold sweep (paraphrase hit rate vs false-positive rate)")
    print(f"{'thr':>5} {'hit_rate':>9} {'false_pos':>10}")
    best = None
    for thr in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        h, f = run(thr)
        print(f"{thr:>5} {h:>9.1%} {f:>10.1%}")
        if f == 0 and (best is None or h > best[1]):
            best = (thr, h)
    if best:
        print(f"\nBest zero-false-positive threshold: {best[0]} (paraphrase hit rate {best[1]:.1%})")
        hit = best[1]
        saved = hit * LLM_LATENCY
        print(f"At a {LLM_LATENCY}s LLM call, expected latency saved per paraphrased request: {saved:.2f}s "
              f"(~{hit:.0%} of LLM calls avoided on repeat-intent traffic)")
    for n in (1_000, 10_000):
        m, p95 = latency(n)
        print(f"Lookup latency @ {n:,} entries: mean {m:.2f} ms, p95 {p95:.2f} ms")
