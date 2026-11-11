#!/usr/bin/env python3
"""Download public primary-source papers and validate them with pdfinfo."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
PAPERS = [
    ('01', 'harness-rsi', '2505.22954v3', 'Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents', 'DGM', 'P1', '自修改代码与开放式候选档案；理解系统级 RSI 的基础。'),
    ('02', 'harness-rsi', '2507.19457v2', 'GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning', 'GEPA', 'P0', '轨迹反思与 prompt 搜索；应纳入低算力对照。'),
    ('03', 'harness-rsi', '2510.04618v3', 'Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models', 'ACE', 'P0', '增量 playbook 更新；记忆或技能演化的重要对照。'),
    ('04', 'harness-rsi', '2607.15524v1', 'Recursive Harness Self-Improvement', 'RHI', 'P1', '直接连接 harness、上下文管理与持续改进。'),
    ('05', 'harness-rsi', '2609.14857v2', 'ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement', 'ModularRSI', 'P0', '模块演化、跨任务对比与 benchmark-disjoint 评估；避免重复。'),
    ('06', 'harness-rsi', '2609.24972v3', 'RRSI: Regularized Recursive Self-Improvement of Agent Harnesses', 'RRSI', 'P0', '约束候选修改与选择，关注泛化和运行成本；使用 v3 而非早期数字。'),
    ('07', 'harness-rsi', '2603.19461v2', 'Hyperagents', 'Hyperagents', 'P1', '允许修改 meta agent 自身；区分改进任务表现与改进改进机制。'),
    ('08', 'harness-rsi', '2609.26457v1', 'Recursive self-improvement of AI research agents', 'AIDE2', 'P2', '研究 Agent 的递归代码改进；了解更大规模闭环。'),
    ('09', 'continual-learning', '2610.25858v1', 'Are Self-Evolving Agents Continual Learners?', 'Self_Evolving_Continual_Learners', 'P0', '直接分析 task drift、全局编辑干扰、遗忘和 retrieval collision；首读。'),
    ('10', 'continual-learning', '2605.06527v1', 'STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?', 'STALE', 'P0', '旧记忆失效和隐式冲突；需区分事实记忆与可执行技能。'),
    ('11', 'continual-learning', '2606.04815v2', 'Learning While Acting: A Skill-Enhanced Test-Time Co-Evolution Framework for Online Lifelong Learning Agents', 'LifeSkill', 'P0', '技能学习与参数内化已有先例；外部技能+RL 并非独立创新。'),
    ('12', 'continual-learning', '2610.29357v1', 'Memory-Centric Continual Learning for AI Agents', 'MCCL', 'P1', '记忆以未来行动收益评价；提供持续学习评估视角。'),
    ('13', 'continual-learning', '2611.05108v1', 'From AutoResearch to Continual AutoResearch: Do Agents Learn Across Tasks?', 'Continual_AutoResearch', 'P1', '持久 Agent 与 reset 对照；区分研究效率和最终解质量。'),
    ('14', 'continual-learning', '2505.11942v3', 'LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners', 'LifelongAgentBench', 'P0', 'DB、OS、KG 交互任务与自动验证；可作为复现实验入口。'),
    ('15', 'small-model-agents', '2602.06485v1', 'AgentCPM-Explore: Realizing Long-Horizon Deep Exploration for Edge-Scale Agents', 'AgentCPM_Explore', 'P0', '4B 主 Agent、训练与上下文问题；pass@64 不等同单次成功率。'),
    ('16', 'small-model-agents', '2506.02153v1', 'Small Language Models are the Future of Agentic AI', 'SLM_Agentic_AI', 'P2', '小模型 Agent 的立场论文；不作为普遍能力优势的实验证明。'),
    ('17', 'agentic-rl', '2604.06268v1', 'RAGEN-2: Reasoning Collapse in Agentic RL', 'RAGEN2', 'P1', '多轮 RL 模板坍塌、信号质量与诊断。'),
    ('18', 'agentic-rl', '2608.17528v1', 'Agent Lightning v1.0: Towards Harnessed Agentic RL', 'Agent_Lightning_v1', 'P0', '实际 harness 参与 RL 的实现与复现入口。'),
    ('19', 'co-evolution', '2609.37105v1', 'VACE: Validation-Gated Alternating Co-Evolution of Agent Models and Harnesses', 'VACE', 'P0', 'RL 与 harness 交替更新+验证门控已经存在；重点排重。'),
    ('20', 'co-evolution', '2609.09134v1', 'Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up Where Imitation Fails', 'On_Policy_Correction', 'P0', '强模型整轨迹模仿可能破坏弱模型与 harness 的匹配。'),
    ('21', 'co-evolution', '2610.36469v1', 'CoHARL: Co-Evolving Agent Policies and Executable Harnesses with Reinforcement Learning', 'CoHARL', 'P0', '共享策略与可执行 harness 群体共演化；比单 harness 交替更接近前沿。'),
    ('22', 'co-evolution', '2506.10943v2', 'Self-Adapting Language Models', 'SEAL', 'P1', '模型生成自身适配数据与指令，再用下游收益训练 self-edit。'),
    ('23', 'co-evolution', '2608.13951v1', 'HELIX: Model-Harness Co-evolution for Recursive Self-Improvement', 'HELIX', 'P1', '干预可追踪、成功与回归轨迹，以及 model-harness-data 循环。'),
    ('24', 'harness-rsi', '2610.14662v1', 'Does Failure Feedback Direct Agent Self-Modification?', 'Failure_Feedback_Self_Modification', 'P0', '区分目标失败修复、相关任务迁移与 held-out 净提升；重要负面证据。'),
    ('25', 'continual-learning', '2609.29545v1', 'ERRAND: Budgeted Maintenance of Agent Memory', 'ERRAND', 'P0', '带预算的记忆重验证已经存在；仅提出何时重查不构成新意。'),
    ('26', 'continual-learning', '2608.15165v2', 'SkillCommit: Evolving Agent Skills through Behaviorally Validated Scope Expansion', 'SkillCommit', 'P0', '局部 patch、跨实例回放与技能提交；首选方向的重要近邻。'),
    ('27', 'continual-learning', '2608.19013v2', 'Harness Continual Learning: Continual Adaptation Beyond Model Parameters', 'HCL', 'P0', 'harness-level forgetting 与更新提交门控已经存在。'),
    ('28', 'continual-learning', '2610.12668v1', 'Recurrent Harness for Agentic Continual Learning', 'ReHarness', 'P0', '严格流式条件下回溯修订已检索经验；需纳入对照。'),
    ('29', 'learning-to-improve', '2607.05297v1', 'MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution', 'MetaSkill_Evolve', 'P0', '两时间尺度演化改进算子；learn-to-improve 方向的直接近邻。'),
    ('30', 'learning-to-improve', '2601.18778v3', 'Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability', 'SOAR_Meta_RL', 'P0', '以学生真实学习增益训练课程生成教师；元奖励已有直接先例。'),
    ('31', 'learning-to-improve', '2606.04703v1', 'Rethinking Continual Experience Internalization for Self-Evolving LLM Agents', 'Continual_Experience_Internalization', 'P0', '多轮内化的能力坍塌、原则级经验与注入方式；参数化持续学习核心。'),
    ('32', 'learning-to-improve', '2610.15492v1', 'When Skills Remain But Access Fails: Factorized Skill Memory for Continual Skill Internalization', 'Factorized_Skill_Memory', 'P1', '区分技能寻址遗忘与执行遗忘，避免把调用失败都当成能力抹除。'),
    ('33', 'learning-to-improve', '2609.35290v2', 'EvoIn: Bridging Evolution and Internalization for Agent Fine-Tuning', 'EvoIn', 'P0', '演化决策流程再内化，撤除 harness 后检验能力保留。'),
    ('34', 'learning-to-improve', '2610.00906v1', 'ActiveSaddler: Automated Curriculum Learning for Agent Harness Optimization', 'ActiveSaddler', 'P1', '课程与 harness 同步演化；简单按失败模式出题已被覆盖。'),
    ('35', 'learning-to-improve', '2610.18807v1', 'SkillScaffold: Scaffolding Agentic Reinforcement Learning with Policy-Adaptive Skill Credit Assignment and Selective Injection', 'SkillScaffold', 'P0', '策略自适应技能注入与无技能推理；harness 促学方向直接近邻。'),
    ('36', 'learning-to-improve', '2609.13739v1', 'HarnessBandit: Joint Learnability-Transferability Scheduling for Multi-Harness Agentic Reinforcement Learning', 'HarnessBandit', 'P0', '联合可学习性与跨 harness 迁移信号调度；避免把跨 harness 训练包装成新题。'),
    ('37', 'learning-to-improve', '2505.03335v3', 'Absolute Zero: Reinforced Self-play Reasoning with Zero Data', 'Absolute_Zero', 'P1', '可执行环境锚定的自生成任务与自博弈。'),
    ('38', 'learning-to-improve', '2511.16043v2', 'Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning', 'Agent0', 'P1', '课程生成与工具执行者共演化。'),
    ('39', 'learning-to-improve', '2610.22029v1', 'Selection Is Not Learning: Choosing Model Futures', 'Selection_Is_Not_Learning', 'P0', '任务看起来可学不代表更新有效；参数继承与经验继承分离。'),
    ('40', 'learning-to-improve', '2610.36826v1', 'When the Critic Falls Behind: Co-Evolving Rubrics and Harnesses for Continual Agent Self-Improvement', 'RuHarness', 'P1', '监督 rubric 与 harness 共演化；验证器方向需进一步越过该近邻。'),
]


def download(item):
    number, category, version, title, short, priority, note = item
    record = dict(number=number, category=category, arxiv_version=version,
                  title=title, priority=priority, note_zh=note,
                  source_url=f'https://arxiv.org/abs/{version}',
                  pdf_url=f'https://arxiv.org/pdf/{version}')
    destination = ROOT / 'papers' / category / f'{number}_{version}_{short}.pdf'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.pdf.part')
    for attempt in range(3):
        try:
            if not destination.exists():
                request = urllib.request.Request(record['pdf_url'], headers={
                    'User-Agent': 'AgentResearchLiteratureDownloader/1.0'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = response.read()
                    record['resolved_url'] = response.url
                if not payload.startswith(b'%PDF-'):
                    raise ValueError('Response is not a PDF')
                temporary.write_bytes(payload)
                check = subprocess.run(['pdfinfo', str(temporary)], capture_output=True,
                                       text=True, timeout=20)
                if check.returncode:
                    raise ValueError('PDF validation failed: ' + check.stderr[:200])
                temporary.replace(destination)
            check = subprocess.run(['pdfinfo', str(destination)], capture_output=True,
                                   text=True, timeout=20, check=True)
            first_page = subprocess.run(['pdftotext', '-f', '1', '-l', '1', str(destination), '-'],
                                        capture_output=True, text=True, timeout=20, check=True)
            record.update(status='downloaded_and_parsed',
                          relative_path=str(destination.relative_to(ROOT)),
                          bytes=destination.stat().st_size,
                          sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                          pages=next((int(line.split(':', 1)[1]) for line in check.stdout.splitlines()
                                      if line.startswith('Pages:')), None),
                          first_page_preview=' '.join(first_page.stdout.split())[:500])
            print(f"OK {number} {version} {record['pages']} pages", flush=True)
            return record
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            record['error'] = f'{type(exc).__name__}: {exc}'
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    record['status'] = 'failed'
    print(f"FAILED {number} {version}: {record['error']}", flush=True)
    return record


def main():
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, paper) for paper in PAPERS]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda entry: entry['number'])
    manifest = dict(research_cutoff='2026-11-09',
                    downloaded_at_utc=datetime.now(timezone.utc).isoformat(), papers=results)
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(total=len(results), successful=sum(r['status'] != 'failed' for r in results),
                          bytes=sum(r.get('bytes', 0) for r in results)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
