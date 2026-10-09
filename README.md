# Concept Screening — Challenge C (Research & Innovation)

A candidate-challenge response that answers one business question with data: **should a consumer
brand portfolio team add behavioural and implicit measures to its concept-screening service, and are
they worth the extra cost and time?**

## The question

Concept screening today rests mainly on two *stated* survey metrics — **Stated_Appeal** and
**Purchase_Intent**. The client wants to know whether adding a **behavioural** measure (a simulated
choice task) and an **implicit** measure (an association task) improves launch decisions, and whether
that improvement justifies their cost and time.

## What the analysis found

- **Behavioural is the genuine value-add.** It is the only measure that generalises out of sample
  (test R² **+0.183** vs survey-only **-0.071**; test ROC-AUC **0.669** vs **0.653**), confirmed in
  two independent model families.
- **Implicit is an open question.** Its in-sample gain does not survive out of sample (test R²
  **-0.150**; ROC-AUC **0.550**, at chance) at the small available sample (n = 70).
- **The value hinges on one number.** Behavioural's decision-saving (+€1,494/concept) is real but
  below its +€3,214 premium at equal error-costs; it pays off **if and only if** a stopped winner's
  forgone profit exceeds ~**1.15×** the launch spend (~€85,698).

**Recommendation:** pilot **Behavioural as standard**, test **Implicit on a subset**, and
**randomise** so the next round of evidence is causal.

## Repository structure

```
data/                 historical_data.csv + data_dictionary.csv (750 concept tests, 2022–2025)
analysis/             the analysis code
  data_utilities.py     shared loader / cleaner (single source of truth)
  exploration.py        exploratory data analysis  -> output/logs/exploration.md
  modelling.py          predictive modelling, propensity/IPW, cost-benefit  -> output/logs/modelling.md
  visualisation.py      matplotlib figures  -> output/figures/
  log_writer.py         Markdown logging helper
output/
  logs/                 exploration.md + modelling.md (the supporting analysis file)
  figures/              PNG figures
  plots/                PDF figures
  plots_svg/            SVG figures
  tables/               PowerPoint table exports
```

## Setup & running

Python **3.11+**, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                # create/refresh the virtualenv from uv.lock
uv run python analysis/exploration.py  # EDA -> output/logs/exploration.md
uv run python analysis/modelling.py    # modelling + cost-benefit -> output/logs/modelling.md + figures
```

Both scripts resolve the project root from their own path, so they can be run from any directory.
They regenerate their Markdown logs (and, for `modelling.py`, the figures) in `output/`.

## Notes

- The two logs (`exploration.md`, `modelling.md`) together form the supporting analysis file; every
  figure and table in the recommendation traces back to them.
- `python-pptx` is included for exporting tables to PowerPoint (`output/tables/`).

