# Prompt / agent-definition evolution log

Records changes to agent prompts (`.claude/agents/*.md`) and CLAUDE.md guidance
made to improve the loop. Agent files are edited only through **agent-architect**
(single-writer rule); this file is the rationale ledger.

Adopt a prompt change **only** when it makes the prompt *shorter*, *reduces
failures*, *makes accept/reject clearer*, or *cuts context/token cost*. Reject
prompt bloat. Keep each CLAUDE.md / agent-file edit ≤20 lines.

## Format

```
### <date> — <agent or CLAUDE.md> — <slug>
- Change: ...
- Objective (which of: shorter / fewer failures / clearer verdict / less context): ...
- Lines added/removed: +X / -Y
- Outcome (if measurable): ...
```

## Entries

### 2026-06-24 — fleet — add 4 perf agents (hybrid)
- Change: created `cuda-researcher`, `perf-profiler`, `harness-optimizer`,
  `experiment-strategist`; CUDA implementation/regression/curation routed to the
  existing fleet (native-optimizer / qa-verifier+core-reviewer / results-analyst).
- Objective: fewer failures (context-isolated GPU research + profiling keep large
  logs out of the parent) + clearer verdict (strategist returns exactly 3).
- Lines added/removed: +4 agent files (~60 lines each); CLAUDE.md ≤20 lines added.
- Outcome: pending first loop iterations.
