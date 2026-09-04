# Project context

Research project on compression × cross-session drift in iBCI decoders.
See README.md and docs/ for full context.

# Rules

- Do NOT run apt install, apt remove, or anything touching /etc without asking.
  This machine's JetPack/CUDA setup is fragile and was painful to get working.
- Do NOT pip install outside the falcon-env venv.
- Do NOT write to docs/reproduction.md or docs/literature.md. Those contain
  only personally-verified results.
- When reporting benchmark numbers, show raw output. Do not summarize or round.
- Hardware details are in docs/hardware.md — do not guess versions.
