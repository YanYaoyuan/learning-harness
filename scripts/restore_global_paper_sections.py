"""Restore the original paper's evidence without reviving obsolete status claims."""

from pathlib import Path
import json
import shutil

from learning_harness.common import file_sha


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Original passage changed; inspect the restoration")
    return text.replace(old, new, 1)


def restore_sections(destination):
    old = Path("docs/paper/latex")
    sections = destination / "sections"
    sections.mkdir(exist_ok=True)
    inputs = {}

    def source(name):
        path = old / "sections" / name
        inputs[str(path)] = file_sha(path)
        return path.read_text()

    related = source("related.tex")
    related = (
        related.replace(
            "Our controlled SFT comparison holds the historical pool fixed and changes which action examples receive a finite training budget.",
            "Our SFT comparison holds each historical pool fixed, compares matched and global selection spaces, and measures the trained student's behavior under a common executor.",
        )
        .replace(
            "while LESS uses optimizer-aware gradient features and target demonstrations~\\citep{xia2024less}.",
            "while LESS uses optimizer-aware gradient features and target demonstrations~\\citep{xia2024less}, which supply the scoring construction instantiated in our global gradient baseline.",
        )
        .replace(
            "Our results delimit one proxy under controlled interactive-agent fine-tuning, rather than refute student-aligned selection in general.",
            "Our complete comparisons separate likelihood-based and gradient-based selection under explicit admissible support, rather than refute student-aligned selection in general.",
        )
        .replace(
            "restrict the current claims to descriptive results on fixed development tasks.",
            "keep development evidence distinct from the separately frozen reserved-task replication and restrict inference to the reported finite panels.",
        )
    )
    (sections / "related-work.tex").write_text(related)

    method = source("method.tex")
    method = method.replace(
        "\\section{A Controlled Experience-Selection Protocol}",
        "\\section{Historical Matched-Selection Construction}",
    )
    method = method.replace(
        "The completed selection experiments use the recorded mean over copies of each content identity,",
        "The historical pre-canonical selection experiments used the recorded mean over copies of each content identity,",
    ).replace(
        "We retain that definition for the historical experiments; an independent uniform-scoring check is reported separately in Section~\\ref{sec:scoring}.",
        "We retain this definition for historical provenance. The completed uniform-scoring diagnostic and canonical retraining are reported in Appendix~\\ref{sec:scoring}; the fresh-panel main comparisons use the canonical scores.",
    )
    prefix = r"""
\paragraph{Historical scope.} This appendix restores the original matched-content design, equations, and algorithm. Its 32-cooling/32-heating panel and four/six-initialization studies are historical; they are not the main 64-placement/64-cleaning panel or its eight-initialization global comparison.
\begin{figure}[htbp]\centering
\resizebox{\linewidth}{!}{\input{figures/historical-architecture}}
\caption{The original historical pipeline, separating the collector, student-view examples, matched selection, paired training, and common post-training executor. It documents the predecessor experiment; the main global-selection architecture is Figure~\ref{fig:architecture}.}
\end{figure}
"""
    method = method.replace("\\label{sec:method}", "\\label{sec:method}\n" + prefix, 1)
    (sections / "historical-method.tex").write_text(method)

    experiments = source("experiments.tex")
    experiments = experiments.replace(
        "\\section{Experiments}", "\\section{Detailed Historical Selection Experiments}"
    )
    experiments = experiments.replace(
        "\\label{sec:experiments}",
        "\\label{sec:experiments}\nThese restored results use the older cooling/heating development panels. They remain separate from the fresh-task main estimates; all original counterexamples and repeats are retained.",
    )
    experiments = experiments.replace("tab:main", "tab:historical-main")
    experiments = replace_once(
        experiments,
        "A separately frozen, 96-model paired retraining study using uniform scores is running; its interim outcomes are excluded from all main-paper tables and figures.",
        "The separately frozen, 96-model uniform-score paired retraining study is now complete. Low NLL minus Random is $-3.516$ points in API and $-3.060$ in Generic; its results are distinguished from the older cached-score comparisons rather than replacing them.",
    )
    (sections / "historical-experiments.tex").write_text(experiments)

    appendix = source("appendix.tex")
    appendix = replace_once(
        appendix,
        "The three completed selection studies train 238 new models and execute 15,232 new policy-evaluation episodes in total.",
        "The first three historical selection studies train 238 models and execute 15,232 new policy-evaluation episodes. The table also restores the subsequently completed canonical replication and fresh-panel matrices; reused tasks and checkpoints are not pooled as independent learning observations.",
    )
    appendix = appendix.replace(
        "Study & New models & New episodes", "Study & New trained models & New episodes"
    )
    marker = r"Stored-checkpoint reevaluation & 0 & 128 & 2 checkpoints, original configurations\\"
    added = r"""Canonical-score replication & 96 & 6,144 & 6 initializations $\times$ 4 recipes, 2 sources\\
Fresh seven-rule development & 224 & 28,800 & 224 students plus an untrained base\\
Global missing baselines & 64 & 8,192 & 2 global rules; 32 gradient students reused\\
"""
    appendix = replace_once(appendix, marker, added + marker)
    appendix = appendix.replace(
        "Evaluation families & 32 cooling + 32 heating tasks, equal weights",
        "Historical evaluation & 32 cooling + 32 heating tasks, equal weights",
    )
    config_marker = r"Environment seed / temperature / batch & 11 / 0 / 8\\"
    appendix = replace_once(
        appendix,
        config_marker,
        r"Fresh development / reserved panels & 64 placement + 64 cleaning groups per panel\\"
        + "\n"
        + config_marker,
    )
    appendix = appendix.replace(
        "The comparison changes the scoring procedure, not the training outcomes; the canonical-score retraining study is separate and ongoing.",
        "The diagnostic changes the scoring procedure, not training outcomes; the completed canonical-score retraining is reported separately below.",
    )
    start = appendix.index("\\smallpara{Ongoing canonical-score replication.}")
    end = appendix.index("\\clearpage", start)
    appendix = (
        appendix[:start]
        + r"""\smallpara{Completed canonical-score replication.}
A separate freeze trains all 96 students: six initialization/dropout seeds, four recipes, two sources, and two policies, at the same 192-update budget and with the same 64 historical tasks. Random indices are unchanged; low-NLL schedules use uniform scores. Both arms are retrained. The completed study consumes 14.350 cumulative GPU-hours, with zero failed models. API Random/Low NLL success is 29.688\%/26.172\%, a $-3.516$ point difference; Generic is 6.836\%/3.776\%, a $-3.060$ point difference. API has six negative initialization means, Generic five negative and one positive; all positive cells are retained. This updates the original manuscript's ``ongoing'' status without relabeling the historical panel as a fresh test.

"""
        + appendix[end:]
    )
    appendix = replace_once(
        appendix,
        "These methods require distinct implementation and cost accounting. No scores from incompatible published benchmark splits are inserted into our result table.",
        "The current main study implements the optimizer-aware LESS-style score and reports its public support and preparation cost. SmartAD's complete-trajectory selection/weighting and Rho-1's reference-token criterion are not reproduced by the raw action-NLL comparator. No scores from incompatible published benchmark splits are inserted into our result table.",
    )
    units = r"""
\smallpara{Selection units and retained complete trajectories.}
The action-level records retain their genuine preceding histories. A provenance audit traces 117 original successful trajectories, but only two API tasks and one Generic task offer multiple distinct, fully retained eligible trajectories. A meaningful complete-trajectory selector therefore needs additional candidate support; renaming action examples as whole trajectories would not implement that comparator. This boundary is retained from the original manuscript.

"""
    appendix = appendix.replace(
        "\\smallpara{Provenance and reproducibility.}",
        units + "\\smallpara{Provenance and reproducibility.}",
        1,
    )
    appendix = appendix.replace(
        "\\begin{tabular}{lrrl}",
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrl}",
        1,
    ).replace("\\end{tabular}", "\\end{tabular}}", 1)
    (sections / "historical-appendices.tex").write_text(appendix)

    for name in ("selection-rows.tex", "initialization-rows.tex"):
        path = old / "generated" / name
        inputs[str(path)] = file_sha(path)
        shutil.copyfile(path, destination / "generated" / name)
    for name in (
        "selection-outcomes.pdf",
        "initialization-effects.pdf",
        "scoring-diagnostic.pdf",
        "webshop-calibration.pdf",
        "control-ladder.tex",
    ):
        path = old / "figures" / name
        inputs[str(path)] = file_sha(path)
        shutil.copyfile(path, destination / "figures" / name)
    path = old / "figures/architecture.tex"
    inputs[str(path)] = file_sha(path)
    shutil.copyfile(path, destination / "figures/historical-architecture.tex")
    inputs["docs/CANONICAL_REPLICATION_RESULTS_2026-11-13.zh-CN.md"] = file_sha(
        "docs/CANONICAL_REPLICATION_RESULTS_2026-11-13.zh-CN.md"
    )
    receipt = dict(
        status="original_evidence_restored_with_current_status",
        restored=[
            "related work",
            "selection equations and algorithm",
            "control ladder and historical architecture",
            "complete historical experiments",
            "inventory/configuration tables",
            "per-seed values and repeats",
            "score sensitivity and replacement table",
            "completed canonical replication",
            "WebShop calibration",
            "source-comparison and trajectory-unit boundaries",
        ],
        original_source_files=inputs,
        obsolete_status_updated=True,
        restoration_script_sha256=file_sha(Path(__file__)),
        historical_and_fresh_panels_separated=True,
        original_manuscript_modified=False,
    )
    (destination / "restoration-evidence.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt
