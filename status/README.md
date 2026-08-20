# DoorMan Teacher Gap — Status Index

This directory is the operational status layer for the ongoing effort to close
the Stage 3 -> Stage 4 teacher-training gap. It does not replace or rewrite the
archived source material or diagnostic evidence elsewhere in this repository.

## Goal

Enable the privileged DoorMan teacher to progress reliably from Stage 3
(open) to Stage 4 (swing), then continue downstream training, while detecting
and preventing local optima and reward hacking such as force/contact spike
farming.

## Roles

| Role | Model / owner | Responsibility |
|---|---|---|
| Coordinator | Root agent | Own the goal, dispatch work, integrate evidence, and decide when to escalate or stop an experiment. |
| Plan agent | GPT-5.6 Sol, extra-high reasoning | Set experimental direction, review complex failure modes, and define decision criteria. |
| Action agent | GPT-5.6 Sol, medium reasoning | Implement and validate code/config changes and execute evaluations. |
| Monitor agent | GPT-5.6 Sol, light reasoning | Periodically inspect training/runtime health and report material changes. |
| Status agent | GPT-5.6 Sol, medium reasoning | Maintain this auditable status layer without changing archived source evidence. |

## Files

- `current_experiments.md` — latest known baseline and ablation state plus the
  next comparison point.
- `risk_and_decision_log.md` — open risks, decisions, and the evidence required
  to resolve them.

## Recording rules

1. Runtime observations, interpretations, and decisions are labeled separately.
2. A startup report is not treated as a training result.
3. Comparisons require the same Stage-3 diagnostic protocol and explicit
   checkpoint identity.
4. Existing evidence under `teacher_training/`, and all content under
   `github/` and `arxiv/`, remains unchanged.
5. Each update records an absolute timestamp (Asia/Shanghai) and cites the
   agent/runtime report from which the state was obtained when no local artifact
   has yet been archived.
6. Once a controlled run passes startup confirmation, its configuration is
   frozen until a planned decision checkpoint. Non-critical improvement ideas
   go to the next-run backlog rather than causing a mid-run restart.

Last status-layer update: **2026-08-19 Asia/Shanghai**.
