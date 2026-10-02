#!/usr/bin/env python3
"""Analyze existing predictions only; never train, calibrate, or invoke tools."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

NAMES = ('q10','q50','q90','q99')
TAUS = np.array([.1,.5,.9,.99])
ALGORITHMS = ('empirical','ewma','cluster','qrf','lightgbm')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def table(headers, body):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(c) for c in row)+' |' for row in body])


def paired_bootstrap(records, pairs, repeats=2000, seed=20260927):
    first=next(iter(records.values()))
    ids=sorted(r['sample_id'] for r in first)
    groups=sorted({r['task_group_id'] for r in first})
    tools=sorted({r['adapter']+'/'+r['tool'] for r in first})
    gi,ti={g:i for i,g in enumerate(groups)},{t:i for i,t in enumerate(tools)}
    counts=np.zeros((len(groups),len(tools)))
    first_map={r['sample_id']:r for r in first}
    for r in first:
        counts[gi[r['task_group_id']],ti[r['adapter']+'/'+r['tool']]]+=1
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=repeats)
    den=weights@counts
    valid=(den>0).all(axis=1);den=den[valid];weights=weights[valid]
    boot={};points={}
    for name,rr in records.items():
        by_id={r['sample_id']:r for r in rr}
        assert sorted(by_id)==ids
        sums=np.zeros((len(groups),len(tools),4))
        for sid in ids:
            r=by_id[sid];base=first_map[sid]
            assert r['y_ms']==base['y_ms'] and r['task_group_id']==base['task_group_id']
            sums[gi[r['task_group_id']],ti[r['adapter']+'/'+r['tool']]] += [r['score']['pinball_ms'][q] for q in NAMES]
        boot[name]=((weights@sums.reshape(len(groups),-1)).reshape(-1,len(tools),4)/den[:,:,None]).mean(axis=1)
        points[name]=(sums.sum(axis=0)/counts.sum(axis=0)[:,None]).mean(axis=0)
    result=[]
    for a,b in pairs:
        delta=boot[a]-boot[b]
        result.append({'a':a,'b':b,'point_macro_mean_delta_ms':float((points[a]-points[b]).mean()),
                       'paired_95_macro_mean_delta_ms':np.quantile(delta.mean(axis=1),[.025,.975]).tolist(),
                       'point_q99_macro_delta_ms':float(points[a][3]-points[b][3]),
                       'paired_95_q99_macro_delta_ms':np.quantile(delta[:,3],[.025,.975]).tolist()})
    return {'unit':'task_group','groups':len(groups),'tools':len(tools),'requested_repeats':repeats,
            'valid_repeats':len(weights),'seed':seed,'difference':'A minus B; negative favors A','comparisons':result}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    run=args.run_dir.resolve();final=Path(str(run)+'_final');control=Path(str(final)+'.autofinalize')
    project=Path(__file__).resolve().parents[1]
    assert (run/'exit_code').read_text().strip()=='0'
    assert (final/'exit_code').read_text().strip()=='0'
    state=read(control/'state.json');assert state['state']=='complete'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    hashes={};tune={};test={};streams={}
    for split,root,names in [('tune',run,[*ALGORITHMS,'ewma_online']),
                             ('test',final,[a+'_test_'+mode for a in ALGORITHMS for mode in ['raw','calibrated']])]:
        for name in names:
            folder=root/name
            metric,manifest=read(folder/'metrics.json'),read(folder/'manifest.json')
            rows=[json.loads(line) for line in (folder/'predictions.jsonl').open()]
            assert manifest['status']=='complete' and not manifest['smoke_only']
            assert manifest['evaluation_split']==split
            assert len(rows)==(694 if split=='tune' else 674)
            y=np.array([r['y_ms'] for r in rows]);q=np.array([[r['prediction']['duration_ms'][n] for n in NAMES] for r in rows])
            assert np.isfinite(q).all() and (q>=0).all() and (np.diff(q,axis=1)>=0).all()
            e=y[:,None]-q;loss=np.maximum(TAUS*e,(TAUS-1)*e)
            np.testing.assert_allclose(loss,[[r['score']['pinball_ms'][n] for n in NAMES] for r in rows])
            np.testing.assert_allclose(loss.mean(axis=0),[metric['micro']['pinball_ms'][n] for n in NAMES])
            np.testing.assert_allclose((y[:,None]<=q).mean(axis=0),[metric['micro']['coverage'][n] for n in NAMES])
            tool_scores={}
            for r in rows:
                tool_scores.setdefault(r['adapter']+'/'+r['tool'],[]).append([r['score']['pinball_ms'][n] for n in NAMES])
            macro=np.array([np.mean(v,axis=0) for v in tool_scores.values()]).mean(axis=0)
            np.testing.assert_allclose(macro,[metric['tool_macro_pinball_ms'][n] for n in NAMES])
            np.testing.assert_allclose(macro.mean(),metric['tool_macro_mean_pinball_ms'])
            entry={'metric':metric,'manifest':manifest}
            (tune if split=='tune' else test)[name]=entry
            if split=='test':streams[name]=rows
            for file in ('metrics.json','manifest.json','predictions.jsonl'):
                hashes[str((folder/file).relative_to(project))]=digest(folder/file)
    calibration={a:read(final/(a+'_calibrated')/'manifest.json') for a in ALGORITHMS}
    for a in ALGORITHMS:
        for folder in (run/a,final/(a+'_calibrated'),control/'selected_models'/a):
            m=read(folder/'manifest.json');assert digest(folder/'model.joblib')==m['model_sha256']
            hashes[str((folder/'model.joblib').relative_to(project))]=m['model_sha256']
    data=read(project/'runs/predictor_prepared/v1/manifest.json')['splits']
    pairs=[('lightgbm_test_raw','empirical_test_raw'),('lightgbm_test_calibrated','empirical_test_raw'),
           ('lightgbm_test_raw','qrf_test_raw'),('lightgbm_test_calibrated','lightgbm_test_raw'),
           ('qrf_test_calibrated','qrf_test_raw')]
    paired=paired_bootstrap(streams,pairs)
    improvements={}
    for split,entries,base in [('tune',tune,'empirical'),('test_raw',test,'empirical_test_raw')]:
        name='lightgbm' if split=='tune' else 'lightgbm_test_raw'
        a,b=entries[name]['metric'],entries[base]['metric']
        improvements[split]={'lightgbm_macro_pinball_reduction_pct':100*(1-a['tool_macro_mean_pinball_ms']/b['tool_macro_mean_pinball_ms']),
                             'lightgbm_micro_mae_reduction_pct':100*(1-a['micro']['q50_mae_ms']/b['micro']['q50_mae_ms'])}
    tail=[]
    for name,rr in streams.items():
        exceed=[r for r in rr if r['y_ms']>r['prediction']['duration_ms']['q99']]
        tail.append({'name':name,'exceedances':len(exceed),'exceedance_groups':len({r['task_group_id'] for r in exceed}),
                     'max_excess_ms':max((r['y_ms']-r['prediction']['duration_ms']['q99'] for r in exceed),default=0)})
    analysis={'input_sha256':hashes,'verified_prediction_rows':6*694+10*674,'paired_bootstrap':paired,
              'improvements':improvements,'tail_events':tail,'calibration_offsets':{a:m['offsets_ms'] for a,m in calibration.items()},
              'no_training_or_inference_performed':True,'analysis_is_posthoc':True}
    (args.output_dir/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')
    sections=[]
    def add(s):sections.append(s)
    def f(x):return f'{x:.3f}'
    def pct(x):return f'{100*x:.2f}%'
    def summary_table(entries,names):
        body=[]
        for name in names:
            m=entries[name]['metric'];u=m['micro']
            body.append([name,f(m['tool_macro_mean_pinball_ms']),f(u['q50_mae_ms']),pct(u['central_80_coverage']),
                         f(u['central_80_width_ms']),pct(u['coverage']['q99']),u['q99_exceedances'],f(u['predict_ms']['p95'])])
        return table(['方法','工具宏平均四分位 pinball↓ / ms','逐调用 Q50 MAE↓ / ms','80% 区间覆盖','区间平均宽度 / ms','Q99 覆盖','Q99 超出数','预测 P95 / ms'],body)
    add('# FlowPilot 五算法实验结果与清理报告（2026-09-27）')
    add('## 1. 完成状态与结论\n\n正式训练、tune 评价、五算法 calibration 与两套冻结 test 均已成功完成，训练和最终阶段退出码均为 0。自动接续状态为 `complete`。本报告仅重算既有预测的统计量与任务组 bootstrap；未重新训练、推理、校准或更改模型。')
    add('- 当前固定配置中，LightGBM 的工具宏平均综合 pinball 最低，且进程内预测较快，是值得继续验证的主要候选。\n- QRF 在 tune 的 Q50 MAE 和部分尾部指标较好，但当前实现预测 P95 约 43 ms；不能满足交接文档提出的 1 ms 初始热路径预算。\n- Q99 尚未稳定达到逐工具 99% 覆盖；LightGBM 的全局校准改善了整体尾部，但明显增加短工具的 Q99 预测开销估计。\n- EWMA 在线适应只在 tune 评估，改善了中位数误差，却恶化了宏平均 Q99 pinball；不能外推为在线 test 或调度收益。\n- 本轮只有一个固定配置和随机种子，未做自动调参、特征消融、神经模型对照、真实 KV 调度或 SLO/goodput 实验。')
    add(table(['阶段','开始 UTC','完成 UTC','耗时'],[
        ['fit＋tune＋EWMA online','17:00:43','17:01:19.724','约 36.7 s（开始时间按运行 ID / tmux 记录）'],
        ['自动接续校准＋test',state['started_at_utc'],state['updated_at_utc'],f((datetime.fromisoformat(state['updated_at_utc'])-datetime.fromisoformat(state['started_at_utc'])).total_seconds())+' s']]))
    add('运行目录：`runs/predictor_experiments/20260927T170043Z/`；最终目录：`runs/predictor_experiments/20260927T170043Z_final/`；接续日志与冻结快照：`runs/predictor_experiments/20260927T170043Z_final.autofinalize/`。')
    add('## 2. 数据与实验协议\n\n目标为单工具客户端 RTT，单位 ms；输出 Q10/Q50/Q90/Q99。下表的组数只统计有有效 RTT 的任务组。原始 fit 清单有 253 个任务组，其中 252 个提供有效 RTT；此前交接中的 253 与这里 252 口径不同。')
    add(table(['分区','有效 RTT','有效 RTT 任务组','用途'],[[s,data[s]['rows'],data[s]['task_groups'],u] for s,u in [('fit','训练基础模型'),('tune','固定配置评价；EWMA 在线对照'),('calibration','拟合全局分位数残差偏移'),('test','冻结模型最终评价')]]))
    add('四分区 task_group 无交集。各评价流全部支持：tune 每流 694/694，test 每流 674/674，共六种工具；没有因 unsupported 排除工具。冻结模式允许使用 T1 前已观测的历史特征，但不在评价标签上更新模型参数。校准是全局 ms 残差偏移，不是逐工具校准，也不具有本任务下严格的覆盖保证。')
    add('当前设置：seed=20260927；每路线程配置 8；QRF 200 棵树；LightGBM 四个分位数头各 200 轮、learning_rate=0.05、max_depth=6、min_samples_leaf=20，标签 log1p；KMeans 最多 8 簇、10 次初始化；EWMA alpha=0.2、残差缓冲 512。没有合并 fit+tune 重训。')
    add('## 3. 指标解释\n\n- 工具宏平均四分位 pinball：先在每种工具内对调用求均值，再对六种工具等权平均，最后对 Q10/Q50/Q90/Q99 等权平均；越低越好。\n- Q50 MAE 是逐调用微平均，不能与工具宏平均混称。\n- 覆盖率按 `y ≤ qτ` 统计；80% 区间为 `[Q10,Q90]`，同时报告宽度。覆盖高于名义值不一定更好，可能来自过宽预测。\n- Q99 超出次数为 `y > Q99`；674 条 test 在名义 99% 下对应约 6.74 次超出，但调用相关，不能当独立二项试验。\n- 预测 P95 包含特征抽取与模型预测，校准结果还包含偏移/重排；不含输入 JSON 解析、状态采集、RPC 和生产调度的全部开销。')
    add('## 4. Tune 结果\n\n'+summary_table(tune,[*ALGORITHMS,'ewma_online']))
    imp=improvements['tune']
    add(f"LightGBM 相比静态经验分布，宏平均 pinball 下降 **{imp['lightgbm_macro_pinball_reduction_pct']:.2f}%**，微平均 Q50 MAE 下降 **{imp['lightgbm_micro_mae_reduction_pct']:.2f}%**。QRF 的 Q50 MAE 更低，但综合 pinball 和预测开销不占优。该结论只描述当前配置，并不证明已完成超参搜索。")
    frozen=tune['ewma']['metric'];online=tune['ewma_online']['metric']
    add(f"EWMA 在线相对冻结：Q50 MAE 下降 {100*(1-online['micro']['q50_mae_ms']/frozen['micro']['q50_mae_ms']):.2f}%，宏平均 pinball 下降 {100*(1-online['tool_macro_mean_pinball_ms']/frozen['tool_macro_mean_pinball_ms']):.2f}%；但 Q99 宏平均 pinball 从 {f(frozen['tool_macro_pinball_ms']['q99'])} 升到 {f(online['tool_macro_pinball_ms']['q99'])} ms。在线更新有收益与尾部代价，不能只汇报 MAE 改善。")
    add('## 5. Test 结果：原始预测\n\n'+summary_table(test,[a+'_test_raw' for a in ALGORITHMS]))
    imp=improvements['test_raw']
    add(f"LightGBM 原始预测相对经验分布：宏平均 pinball 下降 **{imp['lightgbm_macro_pinball_reduction_pct']:.2f}%**，Q50 MAE 下降 **{imp['lightgbm_micro_mae_reduction_pct']:.2f}%**。QRF 在 test 的 Q50 MAE 已不再优于 LightGBM，说明 tune 上局部排序不能直接外推。")
    add('## 6. Test 结果：校准后\n\n'+summary_table(test,[a+'_test_calibrated' for a in ALGORITHMS]))
    add(table(['算法','校准前宏 pinball','校准后宏 pinball','相对变化（负为改善）','Q99 超出数 前→后'],[
        [a,f(test[a+'_test_raw']['metric']['tool_macro_mean_pinball_ms']),f(test[a+'_test_calibrated']['metric']['tool_macro_mean_pinball_ms']),
         f"{100*(test[a+'_test_calibrated']['metric']['tool_macro_mean_pinball_ms']/test[a+'_test_raw']['metric']['tool_macro_mean_pinball_ms']-1):+.3f}%",
         str(test[a+'_test_raw']['metric']['micro']['q99_exceedances'])+' → '+str(test[a+'_test_calibrated']['metric']['micro']['q99_exceedances'])] for a in ALGORITHMS]))
    add('LightGBM 的 80% 区间覆盖由 74.93% 提升到 78.64%，Q99 覆盖由 96.74% 提升到 98.52%，但仍不是稳定的 99% 保证。经验分布的 80% 覆盖由 76.26% 降到 72.11%；QRF 的 80% 覆盖从 84.87% 变为 77.30%，Q99 超出从 8 增到 12。极小的综合指标变化不能夸大成确定收益。')
    add('### 6.1 校准偏移及短工具代价\n\n'+table(['算法','Q10 偏移 / ms','Q50 偏移 / ms','Q90 偏移 / ms','Q99 偏移 / ms'],[[a,*[f(calibration[a]['offsets_ms'][q]) for q in NAMES]] for a in ALGORITHMS]))
    add('LightGBM 的全局 Q99 偏移是 **+804.681 ms**，同样加到短文档读取和文件编辑工具上。该事实解释了整体搜索尾部改善与短工具过度保守同时发生；不能把“总体 Q99 覆盖提高”当作所有工具都变好。以下为 test 中每种工具预测 Q99 的中位数：')
    med=[]
    for tool in sorted(test['lightgbm_test_raw']['metric']['by_tool']):
        values=[]
        for mode in ['raw','calibrated']:
            rr=[r for r in streams['lightgbm_test_'+mode] if r['adapter']+'/'+r['tool']==tool]
            values.append(float(np.median([r['prediction']['duration_ms']['q99'] for r in rr])))
        med.append([tool,*map(f,values)])
    add(table(['工具','原始 Q99 中位数 / ms','校准后 Q99 中位数 / ms'],med))
    add('## 7. 逐工具 Test 分析\n\n下表为各工具四分位平均 pinball（ms），越低越好。任务组在同一 benchmark 的工具之间可能重叠，不能把行内组数相加当独立任务数。')
    tools=sorted(test['empirical_test_raw']['metric']['by_tool'])
    add(table(['工具','行数 / 任务组','经验 raw','QRF raw','LightGBM raw','LightGBM 校准'],[
        [t,str(test['empirical_test_raw']['metric']['by_tool'][t]['rows'])+' / '+str(test['empirical_test_raw']['metric']['by_tool'][t]['task_groups']),
         *[f(test[a]['metric']['by_tool'][t]['mean_pinball_ms']) for a in ['empirical_test_raw','qrf_test_raw','lightgbm_test_raw','lightgbm_test_calibrated']]] for t in tools]))
    add('收益主要来自 search 和 code_terminal。对文件编辑与少样本文档读取，经验分布仍是有竞争力的廉价基线；LightGBM 的全局尾部校准使部分短工具损失增加。不能据 test 的逐工具排名直接拼装一个“每类选最优”的新模型并把本表当其无偏成绩；这种组合需在 tune 制定并用新的未使用留出集验证。')
    add('### 7.1 Q99 逐工具覆盖\n\n'+table(['工具','经验 raw','QRF raw','LightGBM raw','LightGBM 校准'],[
        [t,*[pct(test[a]['metric']['by_tool'][t]['coverage']['q99']) for a in ['empirical_test_raw','qrf_test_raw','lightgbm_test_raw','lightgbm_test_calibrated']]] for t in tools]))
    add('LightGBM 校准后 Hotpot/search 的 Q99 覆盖仍只有 96.63%（3/89 次超出）；BrowseComp/search 为 98.21%（6/336 次超出）。短工具中的 100% 也不是精确保障：例如 get_document 只有 15 条 test，calibration 更只有 4 条。')
    add('## 8. 不确定性与尾部风险\n\n原指标文件包含 500 次 task_group bootstrap。本报告另从既有预测进行 **2000 次配对任务组 bootstrap**，共同重采样 52 个 test 任务组并保持六种工具的等权口径；没有训练或推理。差值为 A−B，负值表示 A 更低。以下区间是本次单划分、单种子上的经验不确定性，不是多重比较校正后的显著性检验。')
    add(table(['A','B','综合宏 pinball 差 / ms','配对 95% 区间 / ms','Q99 宏 pinball 差 / ms','Q99 配对 95% 区间 / ms'],[
        [r['a'],r['b'],f(r['point_macro_mean_delta_ms']),'['+', '.join(map(f,r['paired_95_macro_mean_delta_ms']))+']',f(r['point_q99_macro_delta_ms']),'['+', '.join(map(f,r['paired_95_q99_macro_delta_ms']))+']'] for r in paired['comparisons']]))
    add(table(['方法','整体 Q99 覆盖','原 500 次 bootstrap 95% 区间'],[[name,pct(test[name]['metric']['micro']['coverage']['q99']),'['+', '.join(pct(x) for x in test[name]['metric']['bootstrap']['micro_q99_coverage_95_interval'])+']'] for name in ['empirical_test_raw','qrf_test_raw','lightgbm_test_raw','lightgbm_test_calibrated']]))
    add('配对区间的解读：LightGBM 与经验分布的综合误差差值区间完全低于 0，支持它在本次划分上的改善；LightGBM 与 QRF 的差值区间跨过 0，因此尚不能断言两者精度存在稳定差距。LightGBM 校准前后的差值区间也跨过 0，校准收益仍有不确定性。\n\nQ99 需同时看超出次数、超出幅度与过度保守成本。LightGBM 校准后的宏平均 Q99 pinball 为 34.815 ms，仍高于经验分布原始值 30.909 ms；因此它不是全部指标的优胜者。覆盖区间跨过 99% 也不能证明实现了严格或逐工具 99% 覆盖。')
    add('## 9. 运行开销与适用范围\n\n'+table(['算法','fit 耗时 / s','tune 预测 P95 / ms','顺序 test raw 预测 P95 / ms'],[[a,f(tune[a]['manifest']['fit_seconds']),f(tune[a]['metric']['micro']['predict_ms']['p95']),f(test[a+'_test_raw']['metric']['micro']['predict_ms']['p95'])] for a in ALGORITHMS]))
    add('QRF 在顺序最终测试时仍约 43 ms，因此不能只归因于五路训练争抢资源。这是当前 QRF 实现的测量，不代表所有 QRF 实现的固有下限；逐条森林调用、线程调度和叶节点分布聚合可能有优化空间，需 profiling 验证。其他方法的进程内 P95 本轮低于 1 ms，但还不足以声称完整线上接口达标。GPU 未用于本轮训练。')
    add('本轮未预测整批 next-request-ready，未训练内部 executor 时间独立目标，未验证 hit/follower 路径、KV 迁移能力、正确且按时的 goodput 或端到端加速。不能从 RTT 回归改善推导系统收益。')
    add('## 10. 后续工作建议\n\n1. 保留当前固定配置结果为基线；优先在 fit/tune 做 LightGBM 特征消融与参数搜索，不能再根据本次 test 反复调参并宣称盲测。\n2. 针对 pooled Q99 校准的跨工具偏移，比较分后端/工具的收缩校准或相对尺度残差；calibration 仅 28 组且部分类极少，避免强行细分。\n3. 对 QRF 独立 profiling，比较更低推理线程数、批处理与分布聚合实现，验证预测等价及真正热路径延迟。\n4. 对 EWMA 在线尾部退化检查缓冲、漂移与反馈延迟；在线 test 需另立冻结协议。本轮没有运行它。\n5. 若增加 MLP、冻结文本编码器或 BERT 对照，先用 fit/tune 开发；本 test 已被分析，后续反复利用时需保留新的 untouched holdout。\n6. 调度接入前单独测 KV 能力/成本及 oracle 空间，再评价 SLO 与真实系统收益。')
    add('## 11. 证据与复现\n\n- 正式 tune 汇总：[summary.csv](runs/predictor_experiments/20260927T170043Z/summary.csv)\n- 正式 test 汇总：[summary.csv](runs/predictor_experiments/20260927T170043Z_final/summary.csv)\n- 分析数值、配对区间与输入哈希：[analysis.json](evidence/predictor_results_20260927/analysis.json)\n- 分析脚本：[analyze_time_predictor_results.py](scripts/analyze_time_predictor_results.py)\n- 自动接续状态：[state.json](runs/predictor_experiments/20260927T170043Z_final.autofinalize/state.json)\n- 每个算法的 metrics.json / predictions.jsonl / manifest.json，以及正式模型和冻结快照均保留。\n\n本次重新核对全部 **10904 条** tune/test 预测记录的有限性、非负性、分位数顺序、pinball、覆盖和宏平均；正式模型与冻结快照 SHA256 校验通过。')
    add('复现分析（只读预测明细，不运行模型；使用新的输出文件名）：\n\n```bash\ncd /root/flowpilot_predictor\n.venv-predictor/bin/python scripts/analyze_time_predictor_results.py \\\n  --run-dir runs/predictor_experiments/20260927T170043Z \\\n  --output-dir evidence/predictor_results_recheck \\\n  --report PREDICTOR_EXPERIMENT_REPORT_RECHECK.md\n```')
    args.report.write_text('\n\n'.join(sections)+'\n')
    print(json.dumps({'report':str(args.report),'verified_rows':analysis['verified_prediction_rows'],'improvements':improvements,'paired_bootstrap':paired},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
