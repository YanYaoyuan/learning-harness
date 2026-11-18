#!/usr/bin/env python3
"""Immutable descriptive look at committed matched cells; never changes the GPU plan."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path("runs/variance-diagnostic-v1/stage2")
BASE = Path("output/variance-diagnostic-v1/stage2/interim")
SOURCES = ("api", "generic")
ARMS = ("full_pass", "subset_replay")


def load_snapshot(cutoff):
    freeze = json.loads((ROOT / "freeze.json").read_text())
    cases = {c["name"]: c for c in freeze["cases"]}
    records, provenance = [], {}
    for path in sorted((ROOT / "cases").glob("*/worker-result.json")):
        if path.stat().st_mtime > cutoff:
            continue
        raw = path.read_bytes()
        value = json.loads(raw)
        if not value["complete"] or value["smoke"] or value["case"] != cases[path.parent.name]:
            raise ValueError("Snapshot includes an incomplete or mismatched case")
        for family in ("cool", "heat"):
            summary_path = path.parent / "evaluation" / family / "summary.json"
            summary = json.loads(summary_path.read_text())
            run = json.loads((summary_path.parent / "run.json").read_text())
            if (not summary["complete"] or summary["recorded"] != 32
                    or summary["metrics"]["none"]["error_count"]
                    or summary["metrics"]["none"]["success_rate"] != value["success"][family]
                    or run["revision"] != value["composition_revision"]):
                raise ValueError("Snapshot evaluation audit failed")
        records.append(value)
        provenance[str(path)] = hashlib.sha256(raw).hexdigest()
    return dict(cutoff_unix=cutoff, snapshot_china=datetime.fromtimestamp(cutoff, ZoneInfo("Asia/Shanghai")).isoformat(),
        freeze_sha256=hashlib.sha256((ROOT / "freeze.json").read_bytes()).hexdigest(),
        records=records, provenance=provenance, fixed_main_plan_unchanged=True,
        purpose="User-requested descriptive interim look; no outcome-dependent scheduling or inference")


def analyze(snapshot):
    import numpy as np
    records = { (r["case"]["initialization_seed"], r["case"]["sampling_seed"], r["case"]["source"], r["case"]["arm"]): r
               for r in snapshot["records"] if r["case"]["repeat"] == 0 }
    keys = sorted({k[:2] for k in records if all((*k[:2], s, a) in records for s in SOURCES for a in ARMS)})
    if not keys:
        raise ValueError("No fully paired seed cells")
    results, per_cell, exposure = {}, [], []
    for source in SOURCES:
        group = {}
        for arm in ARMS:
            rows = [records[(*k, source, arm)] for k in keys]
            group[arm] = dict(mean_success=float(np.mean([r["mean_success"] for r in rows])),
                cool_success=float(np.mean([r["success"]["cool"] for r in rows])),
                heat_success=float(np.mean([r["success"]["heat"] for r in rows])),
                mean_nll=float(np.mean([r["mean_nll"] for r in rows])),
                input_tokens=float(np.mean([r["exposure"]["input_tokens"] for r in rows])),
                target_tokens=float(np.mean([r["exposure"]["target_tokens"] for r in rows])))
        deltas = [records[(*k, source, "full_pass")]["mean_success"] - records[(*k, source, "subset_replay")]["mean_success"] for k in keys]
        group["policy_effect"] = dict(success_difference_pp=float(np.mean(deltas)*100),
            wins=sum(x>0 for x in deltas),ties=sum(x==0 for x in deltas),losses=sum(x<0 for x in deltas),
            nll_difference=group["full_pass"]["mean_nll"]-group["subset_replay"]["mean_nll"])
        results[source] = group
    conditional = []
    for init in sorted({k[0] for k in keys}):
        subset = [k for k in keys if k[0] == init]
        row = dict(initialization_seed=init, sampling_seeds=[k[1] for k in subset], complete_four_seed_row=len(subset)==4)
        for arm in ARMS:
            gaps = [100*(records[(*k,"api",arm)]["mean_success"]-records[(*k,"generic",arm)]["mean_success"]) for k in subset]
            row[arm] = dict(gaps_pp=gaps, mean_pp=float(np.mean(gaps)),
                            sampling_sd_pp=float(np.std(gaps,ddof=1)) if len(gaps)>1 else None)
        conditional.append(row)
    for init,sample in keys:
        cell=dict(initialization_seed=init,sampling_seed=sample)
        for source in SOURCES:
            for arm in ARMS:
                record=records[(init,sample,source,arm)]
                cell[source+"_"+arm+"_success"]=record["mean_success"]
                exposure.append(dict(initialization_seed=init,sampling_seed=sample,source=source,arm=arm,
                    **record["exposure"]))
        per_cell.append(cell)
    return dict(interim=True,complete=False,completed_models=len(snapshot["records"]),planned_models=68,
        matched_cells=len(keys),matched_models=len(keys)*4,results=results,
        matched_seed_cells=per_cell,conditional_sampling=conditional,exposure=exposure,
        snapshot_china=snapshot["snapshot_china"],freeze_sha256=snapshot["freeze_sha256"],
        interpretation="Matched means and within-initialization SDs only; fixed development tasks, no calibrated intervals; main plan unchanged")


def create(cutoff):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    name=datetime.fromtimestamp(cutoff,timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=BASE/name
    if out.exists():
        raise ValueError("Interim snapshot is immutable: "+str(out))
    # Recheck all frozen inputs without loading a model or opening new evaluations.
    from run_stage2_sampling import load_freeze
    load_freeze()
    snapshot=load_snapshot(cutoff)
    result=analyze(snapshot)
    out.mkdir(parents=True)
    (out/"inputs.json").write_text(json.dumps(snapshot,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"matched-cells.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result["matched_seed_cells"][0]))
        writer.writeheader()
        writer.writerows(result["matched_seed_cells"])
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),gridspec_kw={"width_ratios":[1,1,1.3]})
    colors={101:"#2878b5",202:"#e58b2a",303:"#59a14f",404:"#b56baf"}
    highest=max(100*cell[s+"_"+a+"_success"] for cell in result["matched_seed_cells"] for s in SOURCES for a in ARMS)
    upper=max(55,float(np.ceil(highest/10)*10+5))
    for ax,source in zip(axes[:2],SOURCES):
        for cell in result["matched_seed_cells"]:
            values=[100*cell[source+"_"+a+"_success"] for a in ARMS]
            ax.plot([0,1],values,"o-",alpha=.7,color=colors[cell["initialization_seed"]],linewidth=1.2,markersize=5)
        means=[100*result["results"][source][a]["mean_success"] for a in ARMS]
        for i,v in enumerate(means):
            ax.plot([i-.13,i+.13],[v,v],color="black",linewidth=3)
            ax.annotate(f"mean {v:.2f}%",(i,v),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9,
                        bbox=dict(facecolor="white",alpha=.85,edgecolor="none",pad=1))
        ax.set_xticks([0,1],["Full pool","Subset replay"])
        ax.set_xlim(-.5,1.5)
        ax.set_ylim(0,upper)
        ax.set_ylabel("Post-A success (%)")
        ax.set_title(source.upper()+f" / {result['matched_cells']} matched seed cells")
        ax.grid(axis="y",alpha=.2)
    ax=axes[2]
    complete_rows=[r for r in result["conditional_sampling"] if r["complete_four_seed_row"]]
    if len(complete_rows)==1:
        row=complete_rows[0]
        for arm,color,style in (("full_pass","#2878b5","o-"),("subset_replay","#e58b2a","s--")):
            sd=row[arm]["sampling_sd_pp"]
            ax.plot(range(4),row[arm]["gaps_pp"],style,color=color,label=f"{arm}: SD {sd:.2f}pp")
        ax.set_xticks(range(4),row["sampling_seeds"])
        ax.set_title(f"Source difference / initialization {row['initialization_seed']}")
        ax.set_xlabel("Sampling seed")
        ax.set_ylabel("API − generic success (pp)")
        ax.legend(fontsize=8)
    elif complete_rows:
        x=np.arange(len(complete_rows))
        for offset,arm,label,color in ((-.18,"full_pass","Full pool","#2878b5"),(.18,"subset_replay","Subset replay","#e58b2a")):
            values=[r[arm]["sampling_sd_pp"] for r in complete_rows]
            bars=ax.bar(x+offset,values,width=.36,label=label,color=color)
            ax.bar_label(bars,fmt="%.2f",fontsize=8,padding=3)
        ax.set_xticks(x,[r["initialization_seed"] for r in complete_rows])
        ax.set_xlabel("Initialization seed (4 sampling seeds each)")
        ax.set_ylabel("Within-initialization source-difference SD (pp)")
        ax.set_title("Source-difference spread by initialization")
        ax.margins(y=.18)
        ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.suptitle(f"INTERIM — {result['completed_models']}/68 models; 768 presentations and 192 updates per new model",fontsize=11)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(out/("matched-policy-interim."+ext),dpi=180)
    plt.close(fig)
    lines=["# Stage 2中途快照", "",snapshot["snapshot_china"],"",
        f"已完成{result['completed_models']}/68个模型，其中{result['matched_models']}个模型构成{result['matched_cells']}组完整配对单元。所有已完成记录均保存在inputs.json；未配齐单元不参与均值比较。", "",
        "两个新组均192次更新、768次示范呈现；同单元使用相同GPU。每个模型评测同一套64个开发任务。", "",
        "| 来源 | 全池遍历成功率 | 子集复用成功率 | 配对差值 | 胜/平/负 |", "|---|---:|---:|---:|---:|"]
    for source in SOURCES:
        r=result["results"][source];e=r["policy_effect"]
        lines.append(f"| {source} | {100*r['full_pass']['mean_success']:.2f}% | {100*r['subset_replay']['mean_success']:.2f}% | {e['success_difference_pp']:+.2f}pp | {e['wins']}/{e['ties']}/{e['losses']} |")
    lines += ["","| 初始化 | 已完成的采样种子数 | 全池来源差SD | 子集来源差SD |","|---|---:|---:|---:|"]
    for row in result["conditional_sampling"]:
        if len(row["sampling_seeds"])>1:
            lines.append(f"| {row['initialization_seed']} | {len(row['sampling_seeds'])} | {row['full_pass']['sampling_sd_pp']:.2f}pp | {row['subset_replay']['sampling_sd_pp']:.2f}pp |")
    lines += ["","这里的SD只描述同一初始化下API−generic差值随采样种子的变化，不是完整4×4的总体方差分解。两来源平均收益和NLL、任务类型与token暴露均保留在summary.json。",
              "","这是用户要求的中途查看；没有新增检验、改变预算、删选模型或启动额外GPU实验。余下初始化继续用于检验当前趋势的稳定性。", "",
              "![中途成对结果](matched-policy-interim.png)",""]
    (out/"RESULTS.zh-CN.md").write_text("\n".join(lines))
    print(json.dumps(dict(directory=str(out),completed_models=result["completed_models"],matched_cells=result["matched_cells"],results=result["results"])),flush=True)
    return out


def upload(directory):
    import wandb
    out=Path(directory)
    if out.resolve().parent != BASE.resolve():
        raise ValueError("Upload must refer to an interim snapshot directory")
    result=json.loads((out/"summary.json").read_text())
    target=json.loads(Path("output/wandb/connection.json").read_text())
    host=target.get("base_url",target.get("host"))
    if host:
        os.environ["WANDB_BASE_URL"]=host
    identity=result["freeze_sha256"][:12]+"-interim-"+out.name
    run=wandb.init(entity=target["entity"],project=target["project"],id=identity,
        name="Stage 2 / INTERIM / "+str(result["matched_cells"])+" paired seed cells",
        group="variance-diagnostic-v1",job_type="descriptive-interim-look",mode="online",resume="allow",
        dir=str(out.absolute()),save_code=False,settings=wandb.Settings(console="off",disable_code=True,
            disable_git=True,save_code=False,disable_job_creation=True,x_disable_stats=True,init_timeout=60))
    try:
        run.log({"interim/paired-policy-figure":wandb.Image(str(out/"matched-policy-interim.png"))})
        run.summary.update(dict(interim=True,final_study_complete=False,matched_seed_cells=result["matched_cells"],
                                completed_models=result["completed_models"],observed_results=result["results"]))
        artifact=wandb.Artifact("stage2-interim-"+out.name,type="descriptive-interim-results")
        for path in out.iterdir():
            if path.is_file() and path.suffix in (".json",".csv",".png",".pdf",".md"):
                artifact.add_file(str(path))
        run.log_artifact(artifact).wait()
        receipt=dict(status="uploaded",run_id=run.id,run_url=run.url,interim=True)
        (out/"archival.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt),flush=True)
    finally:
        run.finish()


def watch():
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/"observer.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan_path=BASE/"observation-plan.json"
    plan=dict(milestones=[8,12,16],initial_manual_look="20261112T080354Z",read_only=True,
              no_outcome_dependent_scheduling=True,teacher_calls=0,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:
            raise ValueError("Observation plan changed; inspect before restarting")
    else:
        plan_path.write_text(json.dumps(plan,indent=2)+"\n")
    state_path=BASE/"observer-state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(status="watching",looks={})
    def persist():
        temporary=state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state,indent=2)+"\n")
        temporary.replace(state_path)
    deadline=time.monotonic()+16*3600
    while True:
        now=time.time()
        snapshot=load_snapshot(now)
        keys={(r["case"]["initialization_seed"],r["case"]["sampling_seed"],r["case"]["source"],r["case"]["arm"])
              for r in snapshot["records"] if r["case"]["repeat"]==0}
        paired={k[:2] for k in keys if all((*k[:2],s,a) in keys for s in SOURCES for a in ARMS)}
        for milestone in plan["milestones"]:
            key=str(milestone)
            if len(paired)>=milestone and key not in state["looks"]:
                out=create(now)
                state["looks"][key]=dict(directory=str(out),actual_paired_cells=len(paired),cloud_status="pending")
                persist()
                with (out/"upload.log").open("w") as log:
                    try:
                        subprocess.run([str(Path(".venv-wandb/bin/python").absolute()),__file__,"--upload",str(out)],
                            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
                        state["looks"][key]["cloud_status"]="uploaded"
                    except Exception as error:
                        state["looks"][key].update(cloud_status="failed",error_type=type(error).__name__)
                persist()
                # A delayed observer may pass several milestones at once; one snapshot
                # records all observed cells, without duplicating it under several names.
                for passed in plan["milestones"]:
                    if passed<=len(paired) and str(passed) not in state["looks"]:
                        state["looks"][str(passed)]=dict(state["looks"][key],covered_by_milestone=milestone)
                persist()
        execution=json.loads((ROOT/"execution.json").read_text())
        if execution["status"] in ("completed","failed","interrupted","stopped_at_cap"):
            state.update(status="finished",main_execution_status=execution["status"])
            persist()
            return
        if time.monotonic()>deadline:
            state["status"]="observer_deadline"
            persist()
            return
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cutoff-unix",type=float)
    mode.add_argument("--upload")
    mode.add_argument("--watch",action="store_true")
    args=parser.parse_args()
    if args.watch:
        watch()
    elif args.upload:
        upload(args.upload)
    else:
        create(args.cutoff_unix)


if __name__=="__main__":
    main()
