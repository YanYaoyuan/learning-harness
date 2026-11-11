#!/usr/bin/env python3
"""Reproducible, vector-exportable figures from completed measured runs only."""
from __future__ import annotations

import csv
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").absolute()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MAIN = Path("runs/dual5090-v2")
AUX = Path("runs/dual5090-v2-filtered")
OUT = Path("output/figures/harness-v2")
SEEDS = [101, 202, 303]
REGIMES = ["raw", "matched", "filtered"]
TITLES = ["Raw successes", "Task/count matched", "Matched + feedback filter"]
SEED_COLORS = ["#4477AA", "#EE7733", "#228833"]
SOURCE_COLORS = ["#4477AA", "#CC6677"]
FAMILIES = ["pick_and_place_simple", "pick_clean_then_place_in_recep",
            "pick_cool_then_place_in_recep", "pick_heat_then_place_in_recep"]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def export_csv(name, records):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save(figure, name):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def paired_figure(result):
    sub = result["fresh_primary"]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.7), sharey=True)
    for ax, regime, title, letter in zip(axes, REGIMES, TITLES, "abc"):
        scores = np.array([sub["cells"][f"{regime}-{arm}"]["wins_per_seed"] for arm in ("none", "early")]).T
        ax.axhline(sub["base_wins"] / 28 * 100, color="#888888", linestyle="--", linewidth=1.1)
        for index, (seed, counts, color) in enumerate(zip(SEEDS, scores, SEED_COLORS)):
            x = np.array([0.0, 1.0]) + (index - 1) * 0.035
            ax.plot(x, counts / 28 * 100, color=color, marker=["o", "s", "^"][index],
                    markersize=7, linewidth=1.8, alpha=0.95)
        for side in (0, 1):
            for count in sorted(set(scores[:, side])):
                ax.annotate(f"{count}/28", (side, count / 28 * 100),
                            xytext=(-15 if side == 0 else 15, 4), textcoords="offset points",
                            ha="right" if side == 0 else "left", fontsize=9, color="#333333")
        ax.set_title(f"({letter}) {title}", loc="left", fontsize=11, pad=12)
        ax.set_xticks([0, 1], ["Unassisted\nrollouts", "Early-hint\nrollouts"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(0, 43)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.6)
        ax.set_xlabel("Training-data source", labelpad=8)
    axes[0].set_ylabel("Unsupported test success (%)")
    handles = [Line2D([0], [0], color=c, marker=m, label=f"Training seed {s}")
               for s, c, m in zip(SEEDS, SEED_COLORS, ["o", "s", "^"])]
    handles.append(Line2D([0], [0], color="#888888", linestyle="--", label="Frozen base (4/28)"))
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False, fontsize=9)
    figure.suptitle("Learning outcomes vary with training seed", fontsize=14, fontweight="bold", y=1.01)
    figure.subplots_adjust(top=0.79, bottom=0.25, wspace=0.18)
    figure.text(0.5, 0.015, "28 unique development tasks; all support removed at evaluation. Lines pair training seeds across data sources, not training steps.", ha="center", fontsize=9)
    save(figure, "01_paired_seeds")


def effects_figure(result):
    figure, ax = plt.subplots(figsize=(9.0, 4.4))
    contrasts = result["fresh_primary"]["contrasts"]
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1.1)
    for index, regime in enumerate(REGIMES):
        y = 2 - index
        c = contrasts[regime]
        mean, (low, high) = c["early_minus_none"] * 100, np.array(c["crossed_bootstrap_95"]) * 100
        ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], color="#26374A", marker="D",
                    markersize=7, linewidth=2.0, capsize=5, zorder=4)
        for shift, color, value, marker in zip([-0.14, 0, 0.14], SEED_COLORS, c["per_seed_delta"], ["o", "s", "^"]):
            ax.scatter(value * 100, y + shift, color=color, marker=marker, s=38, zorder=5)
        ax.text(1.025, y, f"{mean:+.1f} pp\n[{low:+.1f}, {high:+.1f}]", transform=ax.get_yaxis_transform(), va="center", fontsize=10)
    ax.set_yticks([2, 1, 0], TITLES)
    ax.set_xlim(-18, 30)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xticks([-10, 0, 10, 20, 30])
    ax.set_xlabel("Early-hint source minus unassisted source (percentage points)")
    ax.set_title("Estimated source effects remain uncertain", loc="left", fontsize=13, fontweight="bold", pad=18)
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.6)
    handles = [Line2D([0], [0], color="#26374A", marker="D", label="Mean + 95% crossed bootstrap interval")]
    handles.extend(Line2D([0], [0], color=c, marker=m, linestyle="", label=str(s)) for s,c,m in zip(SEEDS,SEED_COLORS,["o","s","^"]))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.28, right=0.79, bottom=0.29, top=0.84)
    figure.text(0.5, 0.015, "Paired resampling over 3 training seeds and 28 task groups. Intervals spanning zero do not establish equivalence.", ha="center", fontsize=8.5)
    save(figure, "02_source_effects")


def collection_figure(main, aux):
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    collection = main["collection"]["metrics"]
    values = [collection[a]["success_rate"] * 100 for a in ("none", "early")]
    names = ["Unassisted", "Early hint"]
    ax = axes[0, 0]
    ax.bar(names, values, color=SOURCE_COLORS, width=0.5)
    for i, v in enumerate(values):ax.text(i, v + 1.2, f"{round(v / 100 * 192)}/192", ha="center", fontsize=10)
    ax.set_ylim(0, 40)
    ax.set_ylabel("Collection success (%)")
    ax.set_title("(a) More successful rollouts", loc="left")
    ax = axes[0, 1]
    counts = [main["data_selection"]["raw_successful_tasks"][a] for a in ("none", "early")]
    ax.bar(names, counts, color=SOURCE_COLORS, width=0.5)
    for i,v in enumerate(counts):ax.text(i, v + 1, str(v), ha="center")
    ax.set_ylim(0, 40)
    ax.set_ylabel("Unique successful training tasks")
    ax.set_title("(b) Different task coverage", loc="left")
    ax = axes[1, 0]
    x = np.arange(3)
    pools = [main["data_selection"]["raw_examples"], main["data_selection"]["matched_examples"], aux["data_selection"]["matched_examples"]]
    for i, arm in enumerate(("none", "early")):
        v = [pool[arm] for pool in pools]
        ax.bar(x + (i - 0.5) * 0.32, v, width=0.32, color=SOURCE_COLORS[i], label=names[i])
        for pos, count in zip(x + (i - 0.5) * 0.32, v):ax.text(pos, count + 14, str(count), ha="center", fontsize=9)
    ax.axhline(256, color="#333333", linestyle=":", linewidth=1.2)
    ax.set_xticks(x, ["Raw", "Matched", "Filtered"])
    ax.set_ylim(0, 1120)
    ax.set_ylabel("Available action examples")
    ax.set_title("(c) Pools differ; updates process 256 samples", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax = axes[1, 1]
    behavior = main["collection_behavior"]
    fractions = [100 * behavior[a]["successful_trajectory_nothing_happens"] / behavior[a]["successful_trajectory_steps"] for a in ("none", "early")]
    ax.bar(names, fractions, color=SOURCE_COLORS, width=0.5)
    for i,arm in enumerate(("none","early")):
        b=behavior[arm];ax.text(i,fractions[i]+0.9,f"{b['successful_trajectory_nothing_happens']}/{b['successful_trajectory_steps']}",ha="center")
    ax.set_ylim(0, 30)
    ax.set_ylabel("Public no-effect feedback (%)")
    ax.set_title("(d) Failed-action labels in successful traces", loc="left", fontsize=10.5)
    for ax in axes.flat:
        ax.grid(axis="y", color="#ECECEC", linewidth=0.6)
        ax.set_axisbelow(True)
    figure.suptitle("Assistance changes the training-data distribution", fontsize=14, fontweight="bold", y=1.01)
    figure.subplots_adjust(hspace=0.45, wspace=0.32, top=0.91, bottom=0.12)
    figure.text(0.5, 0.015, "96 tasks, 2 rollout seeds/source. Counts are descriptive. The filter uses the exact public feedback 'Nothing happens.'; it also rematches examples.", ha="center", fontsize=8.5)
    save(figure, "03_collection_and_data")


def task_figure(type_records):
    matrix = []
    labels = ["Frozen base"]
    for regime in REGIMES:
        labels += [f"{regime.capitalize()} / unassisted", f"{regime.capitalize()} / early hint"]
    for regime,arm in [("base","none")]+[(r,a) for r in REGIMES for a in ("none","early")]:
        values=[]
        for family in FAMILIES:
            subset=[r["success_percent"] for r in type_records if r["regime"]==regime and r["source"]==arm and r["task_type"]==family]
            assert len(subset)==(1 if regime=="base" else 3)
            values.append(np.mean(subset))
        matrix.append(values)
    matrix=np.array(matrix)
    figure, ax=plt.subplots(figsize=(9.3,5.8))
    im=ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(4),["Place","Clean + place","Cool + place","Heat + place"])
    ax.set_yticks(range(7),labels)
    for i in range(7):
        for j in range(4):ax.text(j,i,f"{matrix[i,j]:.1f}%",ha="center",va="center",fontsize=11,color="white" if matrix[i,j]>55 else "#243345")
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=2);ax.tick_params(which="minor",bottom=False,left=False)
    ax.set_title("Most measured gains occur in placement and cleaning",loc="left",fontsize=13,fontweight="bold",pad=16)
    figure.colorbar(im,ax=ax,label="Unsupported success (%)",fraction=0.04,pad=0.04)
    figure.subplots_adjust(left=.26,right=.88,top=.86,bottom=.15)
    figure.text(.5,.025,"7 unique new tasks/type; trained rows average 3 seeds. Base is a single frozen checkpoint. Cell means are descriptive, not independent task replicates.",ha="center",fontsize=8.5)
    save(figure,"04_task_types")


def reward_figure(audit):
    figure,ax=plt.subplots(figsize=(8.6,3.8))
    left=np.zeros(2)
    kinds=[("both_failure","Both fail","#BBBBBB"),("both_success","Both succeed","#4477AA"),("mixed","Different outcomes","#EE7733")]
    for key,label,color in kinds:
        values=np.array([audit["arms"][a]["group_counts"].get(key,0) for a in ("none","early")])
        percent=values/96*100
        ax.barh([1,0],percent,left=left,color=color,height=.5,label=label)
        for y,l,w,c in zip([1,0],left,percent,values):ax.text(l+w/2,y,str(c),ha="center",va="center",color="white" if key=="both_success" else "#222222",fontsize=11)
        left+=percent
    ax.set_yticks([1,0],["Unassisted","Early hint"])
    ax.set_xlim(0,100);ax.set_xlabel("Fraction of 96 training-task groups (%)")
    ax.set_title("More success does not imply more observed within-task variation",loc="left",fontsize=12,fontweight="bold",pad=18)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.27),ncol=3,frameon=False,fontsize=9)
    figure.subplots_adjust(left=.16,bottom=.31,top=.78,right=.97)
    figure.text(.5,.015,"Exactly 2 rollouts/task. Numbers are task counts. This diagnostic does not establish reward variance for larger groups or later policies.",ha="center",fontsize=8.5)
    save(figure,"05_reward_groups")


def optimization_figure(source_records=None):
    figure,axes=plt.subplots(2,3,figsize=(12.4,6.8),sharex=True,sharey=True)
    records=[];paths=[]
    for col,(regime,title) in enumerate(zip(REGIMES,TITLES)):
        root=AUX if regime=="filtered" else MAIN
        for row,arm in enumerate(("none","early")):
            ax=axes[row,col]
            for seed,color,marker in zip(SEEDS,SEED_COLORS,["o","s","^"]):
                path=root/"evaluation"/f"{regime}-{arm}-seed{seed}"/"training.log"
                logged=[]
                if source_records is None:
                    paths.append(path)
                    for line in path.read_text().replace("\r","\n").splitlines():
                        if line.startswith("{") and "'loss':" in line:
                            logged.append(ast.literal_eval(line))
                else:
                    subset=sorted((r for r in source_records if r['regime']==regime and r['source']==arm and r['training_seed']==seed),key=lambda r:r['optimizer_step'])
                    assert [r['optimizer_step'] for r in subset]==[10,20,30,40,50,60]
                    logged=[dict(loss=r['logged_window_loss'],epoch=r['epoch']) for r in subset]
                assert len(logged)==6, f"Unexpected loss logging cadence: {path}"
                steps=[10*(i+1) for i in range(len(logged))]
                for step,entry in zip(steps,logged):
                    assert abs(entry["epoch"]-step/64)<=.006
                    assert np.isfinite(entry["loss"]) and entry["loss"]>=0
                    records.append(dict(regime=regime,source=arm,training_seed=seed,optimizer_step=step,
                                        logged_window_loss=entry["loss"],epoch=entry["epoch"],log_source=str(path)))
                ax.plot(steps,[e["loss"] for e in logged],color=color,marker=marker,markersize=4,linewidth=1.3)
            if row==0:ax.set_title(title,loc="left",fontsize=11)
            if col==0:ax.set_ylabel(("Unassisted source" if row==0 else "Early-hint source")+"\nLogged SFT loss")
            if row==1:ax.set_xlabel("Optimizer updates")
            ax.set_xticks([10,20,30,40,50,60]);ax.set_ylim(bottom=0);ax.grid(color="#E7E7E7",linewidth=.6)
    figure.suptitle("SFT optimization on the current training corpora",fontsize=14,fontweight="bold",y=1.01)
    ymax=max(r['logged_window_loss'] for r in records)*1.12
    for ax in axes.flat:ax.set_ylim(0,ymax)
    handles=[Line2D([0],[0],color=c,marker=m,label=f"Seed {s}") for s,c,m in zip(SEEDS,SEED_COLORS,["o","s","^"])]
    figure.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    figure.subplots_adjust(top=.84,bottom=.14,hspace=.18,wspace=.15)
    figure.text(.5,.015,"Actual logged 10-update window averages; six points per run. Final training ends at 64 updates. These losses do not measure learning speed on new tasks.",ha="center",fontsize=8.5)
    save(figure,"06_training_loss_diagnostic")
    export_csv("training_loss.csv",records)
    return paths,records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "axes.titleweight":"bold","pdf.fonttype":42,"svg.fonttype":"none","savefig.dpi":300})
    for root in (MAIN,AUX):
        for path,expected in read(root/"inputs.sha256.json").items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    main_result, result=read(MAIN/"results.json"), read(AUX/"results.json")
    assert main_result["complete"] and result["complete"]
    fresh=sorted(r["task_id"] for r in rows(MAIN/"fresh-probe.jsonl")); assert len(fresh)==28
    fresh_set=set(fresh)
    score_records=[];type_records=[];episode_paths=[]
    for regime in ["base"]+REGIMES:
        for arm in (["none"] if regime=="base" else ["none","early"]):
            for seed in ([None] if regime=="base" else SEEDS):
                root=AUX if regime=="filtered" else MAIN
                name="base" if regime=="base" else f"{regime}-{arm}-seed{seed}"
                episode_path=root/"evaluation"/name/"episodes.jsonl";episode_paths.append(episode_path)
                episodes=[r for r in rows(episode_path) if r["task_id"] in fresh_set]
                assert len(episodes)==28 and all(not r["support_injected"] for r in episodes)
                wins=sum(r["success"] for r in episodes)
                if regime!="base":assert wins==result["fresh_primary"]["cells"][f"{regime}-{arm}"]["wins_per_seed"][SEEDS.index(seed)]
                score_records.append(dict(regime=regime,source=arm,training_seed=seed,wins=wins,n_unique_tasks=28,success_percent=wins/28*100))
                for family in FAMILIES:
                    cases=[r for r in episodes if r["task_type"]==family];assert len(cases)==7
                    w=sum(r["success"] for r in cases)
                    type_records.append(dict(regime=regime,source=arm,training_seed=seed,task_type=family,wins=w,n_unique_tasks=7,success_percent=w/7*100))
    export_csv("seed_scores.csv",score_records);export_csv("task_type_scores.csv",type_records)
    effects=[]
    for regime,c in result["fresh_primary"]["contrasts"].items():
        effects.append(dict(regime=regime,mean_effect_pp=100*c["early_minus_none"],low95_pp=100*c["crossed_bootstrap_95"][0],high95_pp=100*c["crossed_bootstrap_95"][1],negative_seeds=sum(d<0 for d in c["per_seed_delta"]),n_seeds=3,n_unique_tasks=28))
    export_csv("effects.csv",effects)
    paired_figure(result);effects_figure(result);collection_figure(main_result,result);task_figure(type_records);reward_figure(read(MAIN/"reward-group-audit.json"))
    training_paths,training_records=optimization_figure()
    sources=[MAIN/"results.json",AUX/"results.json",MAIN/"reward-group-audit.json",MAIN/"fresh-probe.jsonl",Path(__file__)]
    sources+=episode_paths+training_paths
    manifest=dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        measured_only=True,primary_unique_tasks=28,training_seeds=SEEDS,output_formats=["png","pdf","svg"],
        runtime_versions=dict(python=sys.version.split()[0],matplotlib=matplotlib.__version__,numpy=np.__version__),
        caveats=["Endpoint comparisons are not learning curves","Three training seeds only","Matched successful-task population is selected","Feedback filtering also rematches examples","Reward audit uses exactly two rollouts per task"])
    (OUT/"figure-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    portable=dict(measured_only=True,source_manifest=manifest,
        main={k:main_result[k] for k in ['collection','data_selection','collection_behavior']},
        result={k:result[k] for k in ['fresh_primary','data_selection']},type_records=type_records,
        score_records=score_records,effect_records=effects,training_records=training_records,
        reward_audit=read(MAIN/'reward-group-audit.json'))
    (OUT/'plot-data.json').write_text(json.dumps(portable,indent=2)+'\n')
    print(OUT)


if __name__=="__main__":
    main()
