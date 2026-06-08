---
weight: -2
---

# Online Cross-Node Data Generation over RDMA (Fork)

> [!IMPORTANT]
> This page documents a **fork-specific** development path maintained on the
> `novitalabs/speculators` `develop` branch. It diverges deliberately from
> upstream `vllm-project/speculators` `main`. If you are using upstream
> Speculators, the offline `.safetensors` data-generation flow described in the
> main docs is what applies to you.

## Summary

Upstream Speculators generates training data **offline**: a vLLM server writes
per-sample hidden states to disk as `.safetensors`, and the trainer reads them
back from a shared directory. This fork keeps an alternative path —
`speculators/data_generation/vllm_hidden_states_generator.py`
(`VllmHiddenStatesGenerator`) — that produces hidden states **in-process**, so
they can be streamed **online, across nodes**, from a dedicated rollout node to
a trainer node over **RDMA** rather than landing on disk.

This page explains why we keep the in-process generator against upstream's
removal of it, and the rationale behind the RDMA-based online transport.

## Background: what upstream changed

Speculators has, historically, always used **disk-based offline transfer**
between data generation and training — there has never been an in-memory
streaming handoff between the two processes in the canonical pipeline. Even in
the `v0.4.x` line, where `VllmHiddenStatesGenerator.generate()` returned tensors
in-process, the canonical pipeline still wrote `data_{idx}.pt` to disk and the
trainer dataloader read every `.pt` from a directory.

Two upstream changes formalized and then hardened the disk-only model:

- **`#353` (`2a1443c`, `v0.4.0-17`, 2026-03-26)** introduced `vllm_client.py`,
  an OpenAI-HTTP client against a standalone vLLM server.
- **`#433` (`8fdee2d`, `v0.4.0-73`, 2026-04-21) "fully deprecate old data
  generation system"** *deleted* `vllm_hidden_states_generator.py` and switched
  the output format from `.pt` to `.safetensors`. This commit is an ancestor of
  `v0.5.0` but **not** of `v0.4.0` / `v0.4.0.1`, so:
  - `v0.4.x` still ships the in-process generator.
  - `v0.5.0` and later no longer ship it.

In the upstream `v0.5.0` flow, a standalone vLLM server (with hidden-state
extraction) writes per-sample `.safetensors` to disk;
`data_generation_offline.py` is a thin async HTTP client that only collects the
`kv_transfer_params["hidden_states_path"]` paths (with resume / retry /
validate); the trainer reads the `.safetensors` offline from a shared directory.

### Upstream's rationale (and why it is reasonable)

The upstream redesign is a sound decision **for upstream's goals**:

- **Decouple data generation from fragile vLLM internal APIs.** The in-process
  generator imports private `vllm.v1.*` internals (scheduler, executor, request,
  KV-cache utilities) that change between vLLM releases. Talking to a standard
  server over HTTP removes that coupling.
- **Independent scaling.** The generation server can be scaled separately from
  the training job.
- **Avoid shipping large tensors over HTTP.** The server writes `.safetensors`
  directly to disk; the client only moves paths.
- **Production-grade offline batch.** Resume / retry / validate semantics, and a
  portable `.safetensors` artifact.

If your workload fits a disk-backed offline pipeline, you should prefer the
upstream path.

## Why this fork keeps `vllm_hidden_states_generator.py`

Our workload does **not** fit a disk-backed offline pipeline. We run continual
fine-tuning of very large draft models (e.g. Kimi K2.6, a ~1T MoE verifier with
an Eagle3-MLA draft) where:

- **Rollout and training live on different nodes.** Rollout/data generation runs
  on a dedicated 8-GPU TP=8 node; the trainer runs on a separate node. The two
  must exchange a continuous stream of fresh hidden states.
- **We want online, on-policy-ish data.** Hidden states should flow to the
  trainer as they are produced, not be staged as a fixed offline corpus. This
  keeps the draft training distribution close to what the verifier is currently
  producing and lets the buffer continuously refill.
- **Disk is the wrong transport at this scale.** Writing every sample's hidden
  states to `.safetensors`, syncing across nodes, and reading back adds
  significant latency and disk/IO pressure for a high-throughput streaming
  workload.

The in-process generator is the **only place** with a live, in-memory hook on
the hidden states at production time. That hook is what makes online cross-node
streaming possible. Upstream's `v0.5.0` deletion removed exactly this hook;
its disk-only model has no in-process point at which to attach a custom
cross-node transport. Migrating to the upstream server-writes-`.safetensors`
model would mean **re-solving cross-node transfer on top of a disk-first
design** — so for this fork we retain the generator.

It also avoids a large rewrite of our surrounding training framework (Camelot),
which is built around an in-process producer → blob-store → trainer flow.

## The RDMA / online cross-node transport

On top of the in-process generator, the surrounding framework wires a
**two-plane** transport between the rollout node and the trainer node:

- **Control plane — Redis Streams.** Sample-ready notifications are published to
  a Redis stream; the trainer consumes them via a consumer group. Stream lag is
  bounded (backpressure) so the producer pauses when the trainer is not draining,
  keeping the two sides coupled without unbounded buffering.
- **Data plane — Mooncake over RDMA.** The actual hidden-state tensors are
  transferred node-to-node over **RDMA** via Mooncake (a master + metadata
  service, with an `rdma` protocol and a named IB device). The trainer hydrates
  each sample's hidden states directly from the producer's memory segment rather
  than from disk.

Concretely, in our topology a TP=8 rollout node generates hidden states with
`VllmHiddenStatesGenerator`, publishes sample-ready events to Redis, and exposes
the tensors through a Mooncake RDMA segment; a separate trainer node consumes the
Redis events and pulls the tensors over RDMA, never round-tripping through a
shared `.safetensors` directory.

This is the one capability the upstream offline model does not provide:
**online cross-node hidden-state streaming**. Disk-backed `.safetensors` files
are never the primary transport path.

## The cost: per-vLLM-version porting

Keeping the in-process generator is not free. It is **deeply coupled to vLLM v1
internals** — not the public API. It imports and drives, among others:

- `vllm.v1.core.sched.scheduler.Scheduler`
- `vllm.v1.executor.multiproc_executor.MultiprocExecutor`
- `vllm.v1.request.Request`
- `vllm.v1.core.kv_cache_utils` (block hasher, KV-cache group helpers)
- `vllm.utils.hashing.get_hash_fn_by_name`

It also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` before any vLLM import (the
default `fork` breaks CUDA re-init for TP>1), installs a
`worker_extension_cls` to capture hidden states, and drives the v1
`scheduler.schedule()` / `executor.collective_rpc()` loop itself.

Because these are private APIs, **every vLLM bump can break the generator, and
the fork hand-ports it each time.** Compatibility with a given vLLM release is
the result of active porting, not natural stability. Evidence from the
generator's own history:

- `aa7298b` "remove eos_token_id from Request() for vLLM 0.17.1 compat" — the
  vLLM 0.17 `Request` signature changed; fixed by dropping the kwarg.
- `bfa4f17` "update generator to support vllm latest main"; `e854faf` "bump
  vllm to 0.12.0"; `block_hasher` support added.
- The current pin constructs `Request(...)` in the vLLM **0.20** form:
  `block_hasher`, `pooling_params=None`, `arrival_time`, and **no**
  `eos_token_id`.

So the `0.17 → 0.20` range did require real fixes; staying current means
re-porting on each upgrade.

## When to use which path

| | Upstream offline (`v0.5.0+`) | This fork (online, RDMA) |
|---|---|---|
| Transport | disk `.safetensors`, shared dir | RDMA (Mooncake) node-to-node |
| Producer | standalone vLLM HTTP server | in-process `VllmHiddenStatesGenerator` |
| Coupling to vLLM | public HTTP API (stable) | private `vllm.v1.*` internals (per-version port) |
| Best for | offline batch, independent scaling, portable artifacts | online cross-node streaming, continual fine-tuning |
| Data freshness | fixed offline corpus | continuously produced, streamed |

**Rule of thumb:** if a disk-backed offline corpus is acceptable, use the
upstream path — it is more robust and lower-maintenance. Choose this fork only
when you need **online cross-node hidden-state streaming** and are prepared to
pay the per-vLLM-version porting cost for the in-process generator.

## Trade-off summary

We bought a capability upstream never had — online cross-node hidden-state
streaming (Redis Streams control plane + Mooncake/RDMA data plane) — at the cost
of binding to a fork that hand-ports a vLLM-v1-internal generator on each vLLM
bump. Upstream `v0.5.0` deleted exactly this generator to escape that coupling,
but its disk-only offline model has no in-process hook for our cross-node
transport. Medium-term, migrating to upstream `v0.5.0` would mean re-solving
cross-node transfer on top of the server-writes-`.safetensors` model; until then
we keep the generator and re-port it as vLLM evolves.
