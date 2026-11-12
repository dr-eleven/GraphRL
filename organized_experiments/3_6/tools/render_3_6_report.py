"""Render the same eleven analysis sections and bordered tables as 3_5."""
from collections import Counter, defaultdict
from pathlib import Path
import ast
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import unicodedata


def render(ctx):
    B, out, cutoff = ctx['B'], ctx['out'], ctx['cutoff']
    rows, groups, perf = ctx['rows'], ctx['groups'], ctx['perf']
    tools, stages = ctx['tools'], ctx['stages']
    retained = [r for r in rows if not r['discarded']]
    dirty = [r for r in rows if r['dirty']]
    for r in rows:
        p = perf.get(r['window'])
        r['optimizer_step'] = r['window'] if r['window'] <= 320 else p['engine_global_step']
        r['updated'] = True if r['window'] <= 320 else p['optimizer_updated']
    assert sum(bool(r['format_errors'][t]) for r in rows for t in stages)==sum(r['format_error'] for r in rows), 'Format-stage attribution mismatch'
    updates = rows[-1]['optimizer_step']
    bins = [(f'{i+1}–{min(i+100, updates)}', [r for r in rows if i < max(1, r['optimizer_step']) <= min(i+100, updates)]) for i in range(0, updates, 100)]
    bins_all = bins + [('合计', rows)]
    parts, table_audits = [], []
    def add(s=''): parts.append(str(s)+'\n')
    def pct(n,d): return f'{100*n/d:.2f}%' if d else '—'
    def ratio(n,d): return f'{pct(n,d)}（{int(n)}/{int(d)}）' if d else '—（0/0）'
    def avg(v):
        v=list(v)
        return statistics.mean(v) if v else 0
    def m(rr,k): return avg(r[k] for r in rr)
    def fmt(v): return f'{v:.4f}' if isinstance(v,float) else str(v)
    def width(s): return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(s))
    def table(headers, body, bordered=True):
        body=[[fmt(x).replace('\n',' ') for x in r] for r in body]
        assert all(len(r)==len(headers) for r in body)
        if not bordered:
            add('| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(x.replace('|','\\|') for x in r)+' |' for r in body))
            return
        widths=[max(width(r[i]) for r in [headers]+body) for i in range(len(headers))]
        border='+'+'+'.join('-'*(w+2) for w in widths)+'+'
        line=lambda r:'| '+' | '.join(x+' '*(w-width(x)) for x,w in zip(r,widths))+' |'
        rendered=[border,line(headers),border,*[line(r) for r in body],border]
        assert len({width(x) for x in rendered})==1
        table_audits.append(dict(columns=len(headers), display_width=width(border), rows=len(body)))
        add('```text\n'+'\n'.join(rendered)+'\n```')
    def chunked(headers, cats, body_fn, chunk=4):
        for i in range(0,len(cats),chunk):
            cc=cats[i:i+chunk]
            table(headers+cc,[body_fn(label,rr,cc) for label,rr in bins_all])
    def section(title, anchor=None): add(('\n<a id="'+anchor+'"></a>\n' if anchor else '\n')+title)
    def gs(rr): return [groups[k] for k in {(r['step'],r['source']) for r in rr}]
    def same(g,k): return max(r[k] for r in g)-min(r[k] for r in g)<1e-6
    def structural_count(rr,t): return sum(bool(r['format_errors'][t]) for r in rr)
    def sc(rr,t,k): return sum(r['stages'][t][k] for r in rr)
    def fcategory(r,t):
        errors=r['format_errors'][t]
        if not errors:return None
        e=errors[0]
        if 'feedback exceeds' in e.lower():return '反馈超上下文预算'
        if 'without EOS' in e or 'token budget' in e:return '触顶未正常结束'
        if t=='prediction':
            if 'tool call' in e.lower():return '预测仍调用工具'
            if 'Expecting value' in e:return '非纯JSON文本'
            if 'escape' in e:return 'JSON非法转义'
            if 'delimiter' in e:return 'JSON缺逗号括号'
            if 'control character' in e:return 'JSON控制字符'
            if 'fields' in e or 'nonempty' in e or 'must be' in e:return '预测字段内容错误'
            return '其他预测格式错误'
        if 'received 0' in e:return '无可解析调用或提前结束'
        if 'Expected exactly one' in e:return '同轮多次调用'
        if 'Expected tool ' in e:return '工具名或顺序错误'
        if 'Node IDs must' in e:return '节点ID非JSON整数'
        if 'Use only the current tool' in e:return '动作类型或字段错误'
        if 'Tool arguments require' in e or 'Invalid tool arguments' in e:return '工具参数字段错误'
        if 'REWRITE requires' in e:return '改写文本为空或类型错'
        if 'analysis' in e or 'Analyze the current' in e:return 'analysis缺失或为空'
        return '其他工具格式错误'
    def ecategory(e):
        if not e:return None
        mapping={'Edge endpoint missing or self-loop':'端点不存在或自环',
                 'Protected, repeated or conflicting edge action':'保护边或重复冲突',
                 'ADD_EDGE requires an absent edge':'添加已存在边',
                 'DEL_EDGE requires a present edge':'删除不存在边',
                 'Cannot delete a missing, protected or repeated node':'删除缺失保护重复节点',
                 'Text edits are restricted to NC target or LC query endpoints':'改写越过允许节点',
                 'At most one REWRITE per target node':'同节点重复改写',
                 'REWRITE must change text; copying original/current text is illegal':'复制当前文本或仅改大小写',
                 'Span text must be a nonempty string':'改写内容为空或类型错',
                 'Node deletion conflicts with another action':'删节点与其他动作冲突',
                 'Edge action conflicts with node deletion':'边与删节点冲突',
                 'Edge actions cannot specify a relation label':'边动作携带relation'}
        return mapping.get(e,'其他执行错误')
    raw_error_map=defaultdict(Counter)
    for r in rows:
        for t in stages:
            if r['format_errors'][t]:raw_error_map[t+':'+fcategory(r,t)][r['format_errors'][t][0]]+=1
        for t in tools:
            if r['stages'][t]['error']:raw_error_map[t+':'+ecategory(r['stages'][t]['error'])][r['stages'][t]['error']]+=1
    (out/'error_category_mapping.json').write_text(json.dumps(raw_error_map,ensure_ascii=False,indent=2))
    section('# MRRA_B_4Tool：3_6 四工具 7B 实验分析（训练、合并与策略测试）')
    add(f'生成时间：{ctx["now"].isoformat()}。固定统计至数据窗口 **{cutoff}**（row.step=0..{cutoff-1}），实际参数更新 **{updates}** 次。模板为 `3_5/reports/3_5_updates.md`：11节、同类统计、带边框等宽表和真实NC逐轮输入输出。')
    add('主表按每100次实际参数更新分段；整批丢弃窗口按其结束时的累计更新数归入对应段，并列明数据窗口映射。所有“候选”默认含已生成但丢弃的候选；实际训练候选另表。阶段错误归档为互斥中文类别，完整原始报错留在JSON，不将动态输出文字展开成数百列表格。')
    section('## 导航')
    add('\n'.join('- ['+name+'](#'+anchor+')' for anchor,name in [('scope','统计范围和设置'),('findings','主要结论'),('generation','生成和阶段检查'),('legality','动作合法性和命中'),('reward','奖励和恢复质量'),('phases','任务与阶段构成'),('advantages','优势和耗时'),('interpretation','阶段对照与最终测试'),('examples','NC案例索引'),('transcripts','逐轮实录'),('evidence','证据与校验')]))
    section('## 1. 统计范围和实验设置','scope')
    table(['项目','记录'],[['完成数据窗口',cutoff],['实际参数更新',updates],['源组 / 生成候选 / 保留候选',f'{len(groups)} / {len(rows)} / {len(retained)}'],['NC',f'{len(gs([r for r in rows if r["task"]=="NC"]))}源组 / {sum(r["task"]=="NC" for r in rows)}候选'],['LC',f'{len(gs([r for r in rows if r["task"]=="LC"]))}源组 / {sum(r["task"]=="LC" for r in rows)}候选'],['模型','Qwen2.5-7B-Instruct；LoRA r32/alpha32'],['当前训练配置','GPU1 A6000；ZeRO-2无卸载；batch12/accum1/G6'],['优化','lr2e-6；warmup55；cosine；beta0.01；1 epoch计划'],['预算','每轮512；总completion8704含反馈；训练窗口321起总长<5120'],['奖励','S五阶段各0.2；B四阶段各0.25；H结构命中0.1/改写0.2；T=1/0.1/0']],False)
    add('训练在用户要求下提前停止；本报告的checkpoint测试不冒称完整1 epoch结果。原运行续训边界40、320保持不变；BGE修复前历史H不重算，长度丢弃不是预测错误。')
    table(['使用记录','row.step区间','候选'],[[Path(x['path']).parents[1].name,f'{x["step_range"][0]}..{x["step_range"][1]}',x['included_rows']] for x in ctx['manifest']],False)
    table(['实际更新段','对应数据窗口','生成候选','保留候选','整批无更新窗口'],[[lab,f'{min(r["window"] for r in rr)}–{max(r["window"] for r in rr)}',len(rr),sum(not r['discarded'] for r in rr),len({r['window'] for r in rr if not r['updated']})] for lab,rr in bins_all])
    section('## 2. 主要结论','findings')
    add(f'1. S=1为{ratio(sum(abs(r["S"]-1)<1e-6 for r in rows),len(rows))}；有格式终止{sum(r["format_error"] for r in rows)}条，执行非法轨迹{ratio(sum(r["illegal"] for r in rows),len(rows))}。')
    add(f'2. 全量训练候选正确率{ratio(sum(r["correct"] for r in rows),len(rows))}；保留候选正确率{ratio(sum(r["correct"] for r in retained),len(retained))}。T均值包含候选内错误的0.1，不等于正确率。')
    add('3. 奖励均值：'+'、'.join(f'{k}={m(rows,k):.4f}' for k in ['S','B','T','H','R'])+'。B奖励需求与非空选择，H为局部正向命中，都不等于整体图正确。')
    add(f'4. dirty文本+结构完整恢复{ratio(sum(r["full"] for r in dirty),len(dirty))}；clean结构变差{ratio(sum(not r["dirty"] and r["dist_after"]>0 for r in rows),sum(not r["dirty"] for r in rows))}。')
    add(f'5. 保留组R全同{sum(same(g,"R") for g in gs(retained))}/{len(gs(retained))}；丢弃候选{ratio(len(rows)-len(retained),len(rows))}。正式测试结论单列8.5，不能用训练正确率替代。')
    section('## 3. 每100次参数更新的生成和四工具完成情况','generation')
    table(['更新','候选']+[x for t in tools for x in [t+'结构错',t+'图非法']]+['prediction结构错','任一阶段出错轨迹占比'],[[lab,len(rr)]+[x for t in tools for x in [structural_count(rr,t),sc(rr,t,'illegal')]]+[structural_count(rr,'prediction'),ratio(sum(r['format_error'] or r['illegal'] for r in rr),len(rr))] for lab,rr in bins_all])
    add(f'格式错{sum(r["format_error"] for r in rows)}、图非法{sum(r["illegal"] for r in rows)}、两者重叠{sum(r["format_error"] and r["illegal"] for r in rows)}。未到达阶段不反复记失败；反馈预算不足单独归类。')
    section('### 3.1 检查内容及各阶段具体报错数')
    section('#### 3.1.1 当前实际检查哪些内容')
    add('S：每轮恰好一个当前工具、analysis/actions字段、非空analysis、actions列表、op/字段/整数ID及最终严格JSON；图执行：保护对象、节点存在性、自环、重复、边存在性、改写限制。格式错终止且保留此前S/B/H，T=0；图非法整轮原子拒绝，继续下一阶段。')
    section('#### 3.1.2 各阶段结构检查报错明细')
    for t in stages:
        add('**'+t+'阶段**')
        cats=sorted({fcategory(r,t) for r in rows if r['format_errors'][t]})
        chunked(['更新','结构错误量'],cats,lambda lab,rr,cc:[lab,structural_count(rr,t)]+[sum(fcategory(r,t)==c for r in rr) for c in cc])
        if not cats:add('本次无此类报错。')
    section('#### 3.1.3 各阶段图编辑合法性报错明细')
    for t in tools:
        add('**'+t+'阶段**')
        cats=sorted({ecategory(r['stages'][t]['error']) for r in rows if r['stages'][t]['error']})
        chunked(['更新','图非法量'],cats,lambda lab,rr,cc:[lab,sc(rr,t,'illegal')]+[sum(ecategory(r['stages'][t]['error'])==c for r in rr) for c in cc],3)
    add('分类列互斥；分块表重复的总量不能相加。原始错误到中文类别的映射保存在error_category_mapping.json，其他类别不会丢弃原文。')
    section('#### 3.1.4 阶段到达数与调用次数，解释未检查的阶段')
    for t_idx,t in enumerate(stages):
        table(['更新',t+'到达',t+'未到达'],[[lab,sum(r['generated_count']>t_idx for r in rr),sum(r['generated_count']<=t_idx for r in rr)] for lab,rr in bins_all])
    table(['更新']+[str(i)+'次有效派发' for i in range(5)],[[lab]+[sum(sum(r['stages'][t]['dispatch'] for t in tools)==i for r in rr) for i in range(5)] for lab,rr in bins_all])
    section('#### 3.1.5 达到max token仍未正常结束与长度丢弃')
    table(['更新']+[t+'触顶' for t in stages]+['有预算停止','丢弃候选'],[[lab]+[sum(any(e.get('stage')==t for e in r['limit_events']) for r in rr) for t in stages]+[sum(bool(r['limit_events']) for r in rr),sum(r['discarded'] for r in rr)] for lab,rr in bins_all])
    add('触顶事件、格式终止和训练丢弃不是同一指标。512是单轮预算，训练5120是完整序列筛选，8704含外部反馈；恰好输出EOS不记异常。')
    section('### 3.2 各生成阶段的token分布')
    add('仅保留候选有可靠原始token；整组丢弃后的占位token不能作为真实长度。按completion_mask与tool_mask均为1的生成段读取；不含反馈与padding。P90沿用3_5最近秩定义ceil(0.9*n)，而非旧3_6的线性插值。')
    for i,t in enumerate(stages):
        add('**'+t+'阶段**');body=[]
        for lab,rr in bins_all:
            v=sorted(r['tokens'][i] for r in rr if r['token_available'] and len(r['tokens'])>i)
            body.append([lab,len(v),f'{avg(v):.2f}',f'{statistics.median(v):.2f}' if v else '—',v[math.ceil(.9*len(v))-1] if v else '—',max(v) if v else '—'])
        table(['更新','输出段数','平均token','中位数','P90','最大'],body)
    section('### 3.3 最终输出格式与候选标签')
    table(['更新','预测格式有效','T=1正确','T=0.1候选内错','T=0','正确率'],[[lab,sum(r['prediction_format_valid'] for r in rr),sum(r['T']==1 for r in rr),sum(abs(r['T']-.1)<1e-6 for r in rr),sum(r['T']==0 for r in rr),ratio(sum(r['correct'] for r in rr),len(rr))] for lab,rr in bins_all])
    section('## 4. 每100次参数更新的动作合法性和错误位置','legality')
    section('### 4.1 四工具分别统计合法空操作、合法非空和阶段失败')
    for t in tools:
        add('**'+t+'工具**')
        table(['更新','实际派发','合法空操作','合法非空','阶段失败','失败率'],[[lab]+[sc(rr,t,k) for k in ['dispatch','empty','nonempty','illegal']]+[ratio(sc(rr,t,'illegal'),sc(rr,t,'dispatch'))] for lab,rr in bins_all])
    section('### 4.2 按错误类型分段统计')
    table(['更新','仅格式终止','仅图非法','两者重叠','任一错误','无上述错误'],[[lab,sum(r['format_error'] and not r['illegal'] for r in rr),sum(r['illegal'] and not r['format_error'] for r in rr),sum(r['illegal'] and r['format_error'] for r in rr),sum(r['illegal'] or r['format_error'] for r in rr),sum(not(r['illegal'] or r['format_error']) for r in rr)] for lab,rr in bins_all])
    add('3.1.2/3.1.3提供各工具的逐类分段表，本节按轨迹错误位置去重。合法性检查不读取clean参考，非法阶段的提交动作不计已执行动作。')
    section('### 4.3 已提交动作与隐藏clean参考比较')
    add('与3_5同口径：删除节点不在clean；删除边不在clean；添加边在clean；rewrite精确匹配clean文本。H仅针对原始噪声，rewrite允许语义阈值，因此参考匹配和H命中分开。')
    for t in tools:
        add('**'+t+'：全部已生成候选**')
        table(['更新','提交动作','实际执行','满足clean参考','参考比例','日志H命中','H/执行'],[[lab]+[sc(rr,t,k) for k in ['submitted','executed','reference']]+[ratio(sc(rr,t,'reference'),sc(rr,t,'executed')),sc(rr,t,'hit'),ratio(sc(rr,t,'hit'),sc(rr,t,'executed'))] for lab,rr in bins_all])
    table(['工具','保留候选执行','clean参考','原始噪声H命中','H/执行','动态可执行GT'],[[t]+[sc(retained,t,k) for k in ['executed','reference','hit']]+[ratio(sc(retained,t,'hit'),sc(retained,t,'executed')),sc(retained,t,'gt')] for t in tools])
    add('动态GT来自阶段入口的actionable_count，不是原始噪声统一召回分母。长度停止可能发生在动作提交后、反馈尚不可见时，H按历史计分字段保留；对照全部与保留候选，不能把H=0全解释为动作错误。')
    section('## 5. 每100次参数更新的reward和恢复质量','reward')
    section('### 5.1 奖励均值及组内信号')
    for cohort in ['全部生成','实际保留']:
        add('**'+cohort+'候选**');body=[]
        for lab,rr in bins_all:
            ss=rr if cohort=='全部生成' else [r for r in rr if not r['discarded']]
            gg=gs(ss)
            body.append([lab,len(ss)]+[m(ss,k) for k in ['S','B','T','H','R']]+[sum(same(g,'R') for g in gg),ratio(sum(same(g,'R') for g in gg),len(gg))])
        table(['更新','候选','S均值','B均值','T均值','H均值','R均值','全组同分','占源组比例'],body)
    for prefix,tt in [('S',stages),('B',tools),('H',tools)]:
        table(['更新']+[prefix+'_'+t for t in tt],[[lab]+[avg(r['stageS'][t] if prefix=='S' else r['stages'][t]['B'] if prefix=='B' else r['stages'][t]['hit']*(.2 if t=='rewrite' else .1) for r in rr) for t in tt] for lab,rr in bins_all])
    table(['更新','格式错轨迹','其中非零奖励','格式错平均S','格式错平均B','格式错平均H'],[[lab,sum(r['format_error'] for r in rr),sum(r['format_error'] and r['R']>0 for r in rr)]+[m([r for r in rr if r['format_error']],k) for k in ['S','B','H']] for lab,rr in bins_all])
    section('### 5.2 奖励分布与各分项的组内差异')
    vals=sorted({round(r['R'],4) for r in rows})
    for j in range(0,len(vals),10):
        vv=vals[j:j+10]
        table(['更新']+[str(x) for x in vv],[[lab]+[sum(abs(r['R']-x)<1e-5 for r in rr) for x in vv] for lab,rr in bins_all])
    table(['更新','源组']+[k+'组内有差异' for k in ['S','B','T','H']]+['R全同'],[[lab,len(gs(rr))]+[sum(not same(g,k) for g in gs(rr)) for k in ['S','B','T','H']]+[sum(same(g,'R') for g in gs(rr))] for lab,rr in bins_all])
    section('### 5.3 clean/dirty的行为分与B失败原因')
    table(['更新','dirty候选','dirty B=1','clean候选','clean B=1'],[[lab,sum(r['dirty'] for r in rr),sum(r['dirty'] and r['B']>=.999 for r in rr),sum(not r['dirty'] for r in rr),sum(not r['dirty'] and r['B']>=.999 for r in rr)] for lab,rr in bins_all])
    for t in tools:
        add('**'+t+'阶段**');body=[]
        for lab,rr in bins_all:
            ss=[r['stages'][t] for r in rr];valid=[s for s in ss if s['dispatch'] and not s['illegal']]
            body.append([lab]+[sum(s['need']==n and s['has']==h for s in valid) for n,h in [(True,True),(True,False),(False,False),(False,True)]]+[sum(s['illegal'] for s in ss),sum(not s['dispatch'] for s in ss),avg(s['B'] for s in ss)])
        table(['更新','需要且非空','需要却空','无需且空','无需却非空','图非法','未派发','阶段B均值'],body)
    section('### 5.4 实际图恢复结果')
    table(['更新','全部结构exact','全部文本+结构exact','dirty候选','dirty结构exact','dirty文本+结构exact','dirty B=1','其中结构仍错'],[[lab,ratio(sum(r['structure_exact'] for r in rr),len(rr)),ratio(sum(r['full'] for r in rr),len(rr)),sum(r['dirty'] for r in rr),ratio(sum(r['dirty'] and r['structure_exact'] for r in rr),sum(r['dirty'] for r in rr)),ratio(sum(r['dirty'] and r['full'] for r in rr),sum(r['dirty'] for r in rr)),sum(r['dirty'] and r['B']>=.999 for r in rr),sum(r['dirty'] and r['B']>=.999 and not r['structure_exact'] for r in rr)] for lab,rr in bins_all])
    table(['对象','候选','距离改善','变差','不变','初始距离总和','最终距离总和'],[[lab,len(rr),sum(r['dist_after']<r['dist_before'] for r in rr),sum(r['dist_after']>r['dist_before'] for r in rr),sum(r['dist_after']==r['dist_before'] for r in rr),sum(r['dist_before'] for r in rr),sum(r['dist_after'] for r in rr)] for lab,rr in [('全部',rows),('dirty',dirty),('clean',[r for r in rows if not r['dirty']]),('实际保留',retained)]])
    add('沿用3_5的无向端点集合及全部节点文本定义；不验证relation/方向，exact不能解释为完整知识图谱恢复。距离是节点与无向边集合对称差，不包含文本语义。')
    section('### 5.5 3_6新增：rewrite语义评分与H')
    body=[]
    for lab,rr in bins_all:
        pairs=[]
        for r in rr:
            mm=r['metrics']
            for k,v in mm.items():
                if k.endswith('_similarity_before'):
                    prefix=k[:-6];pairs.append((v,mm[prefix+'after'],mm[prefix+'gain']))
        body.append([lab,len(pairs)]+[avg(p[i] for p in pairs) for i in range(3)]+[sum(p[2]>0 for p in pairs),sum(p[2]<0 for p in pairs),sum(p[1]>=.85 and p[2]>=.05-1e-12 for p in pairs)])
    table(['更新','有评分节点','before均值','after均值','gain均值','提升节点','下降节点','双阈值达标'],body)
    add('只使用日志节点评分；未评分不填0；H允许精确恢复例外，故双阈值达标数不必等于全部rewrite H。')
    section('## 6. 每100次参数更新的任务构成及分任务变化','phases')
    section('### 6.1 阶段数据构成')
    table(['更新','源组','候选','NC占比','dirty占比','S','B','T','H','R'],[[lab,len(gs(rr)),len(rr),pct(sum(r['task']=='NC' for r in rr),len(rr)),pct(sum(r['dirty'] for r in rr),len(rr))]+[m(rr,k) for k in ['S','B','T','H','R']] for lab,rr in bins_all])
    section('### 6.2 NC与LC分别比较')
    for task in ['NC','LC']:
        add('**'+task+'**');body=[]
        for lab,rr in bins_all:
            ss=[r for r in rr if r['task']==task]
            body.append([lab,len(gs(ss)),len(ss)]+[m(ss,k) for k in ['S','B','T','H','R']]+[ratio(sum(r['correct'] for r in ss),len(ss)),ratio(sum(r['illegal'] for r in ss),len(ss))])
        table(['更新','源组','候选','S','B','T','H','R','正确率','执行错误率'],body)
    section('### 6.3 各阶段数据集构成（源组数）')
    datasets=sorted({r['dataset'] for r in rows})
    table(['更新']+datasets,[[lab]+[len(gs([r for r in rr if r['dataset']==d])) for d in datasets] for lab,rr in bins_all])
    add('degree顺序使NC/LC、数据集、dirty比例随阶段改变；训练趋势不能直接当作固定样本能力曲线。')
    section('## 7. 组内优势、loss和时间','advantages')
    logs={}
    for name,lo,hi in [(ctx['names'][0],0,40),(ctx['names'][1],40,320),(ctx['names'][2],320,cutoff)]:
        for line in (ctx['RUNS']/name/'train.log').read_text(errors='replace').splitlines():
            i=line.find("{'loss':")
            if i<0:continue
            try:d=ast.literal_eval(line[i:]);w=round(float(d['epoch'])*2747.5)
            except (ValueError,SyntaxError,KeyError):continue
            if lo<w<=hi:logs[w]=d
    def times(rr):
        ww={r['window'] for r in rr}
        return [perf[w]['seconds'] if w in perf else float(logs[w]['step_time']) for w in ww if w in perf or (w in logs and 'step_time' in logs[w])]
    body=[]
    for lab,rr in bins_all:
        ll=[logs[w] for w in {r['window'] for r in rr} if w in logs]
        body.append([lab,len({r['window'] for r in rr}),len(ll),avg(abs(r['advantage']) for r in rr if not r['discarded'] and r['advantage'] is not None)]+[avg(float(d[k]) for d in ll if k in d) for k in ['loss','grad_norm','kl']]+[avg(times(rr))])
    table(['更新','数据窗口','loss记录','保留|adv|均值','loss均值','grad_norm','KL','秒/窗口'],body)
    body=[]
    for lab,rr in [('全部窗口',rows),('最近20个窗口',[r for r in rows if r['window']>cutoff-20])]:
        v=sorted(times(rr));body.append([lab,len(v),avg(v),statistics.median(v) if v else 0,v[math.ceil(.9*len(v))-1] if v else 0,f'{min(v):.2f}/{max(v):.2f}' if v else '—'])
    table(['时间口径','记录数','平均秒','中位秒','P90秒','最小/最大秒'],body,False)
    add('时间为窗口实际记录，含整批丢弃；loss日志用epoch×2747.5映射窗口，避免将窗口误认梯度更新。未记录各轮wall-clock，token数不能直接当轮耗时；小loss也不证明编辑正确。')
    section('## 8. 对当前行为的解释和后续验证重点','interpretation')
    section('### 8.1 本次约束与H的实际表现')
    add(f'保留候选加边满足clean参考{sc(retained,"add_edge","reference")}次，命中原始缺边{int(sc(retained,"add_edge","hit"))}次；二者差异反映补回先前删除边的行为。执行合法、参考匹配和原始噪声命中必须分开。')
    def range_table(ranges):
        body=[]
        for lab,lo,hi in ranges:
            rr=[r for r in rows if lo<=r['optimizer_step']<=hi];gg=gs(rr)
            body.append([lab,len(gg),len(rr),pct(sum(r['task']=='NC' for r in rr),len(rr)),pct(sum(r['dirty'] for r in rr),len(rr))]+[m(rr,k) for k in ['S','B','T','H','R']]+[ratio(sum(r['correct'] for r in rr),len(rr)),sum(r['format_error'] for r in rr),sum(r['illegal'] for r in rr),sum(r['dirty'] and r['full'] for r in rr)])
        table(['更新范围','源组','候选','NC占比','dirty占比','S','B','T','H','R','正确率','格式错','图非法','dirty完整恢复'],body)
    section('### 8.2 第501次更新之后与前500次更新对照')
    range_table([('1–500',1,500),(f'501–{updates}',501,updates),(f'最近100',max(1,updates-99),updates)])
    body=[]
    for lab,lo,hi in [('前500',1,500),('501–截止',501,updates)]:
        rr=[r for r in rows if lo<=r['optimizer_step']<=hi]
        for t in tools:
            body.append([lab,t]+[sc(rr,t,k) for k in ['empty','nonempty','illegal']]+[ratio(sc(rr,t,'illegal'),sc(rr,t,'dispatch')),sum(r['stages'][t]['dispatch'] and not r['stages'][t]['illegal'] and r['stages'][t]['need'] and not r['stages'][t]['has'] for r in rr),sum(r['stages'][t]['dispatch'] and not r['stages'][t]['illegal'] and not r['stages'][t]['need'] and r['stages'][t]['has'] for r in rr)])
    table(['范围','工具','合法空','合法非空','图非法','派发拒绝率','需要却空','无需却非空'],body)
    section('### 8.3 第1101次更新之后的新增分析')
    range_table([('1–1100',1,1100),(f'1101–{updates}',1101,updates)])
    table(['范围','任务','候选','正确率','H均值'],[[lab,task,len(rr),ratio(sum(r['correct'] for r in rr),len(rr)),m(rr,'H')] for lab,lo,hi in [('前1100',1,1100),('1101–截止',1101,updates)] for task in ['NC','LC'] for rr in [[r for r in rows if lo<=r['optimizer_step']<=hi and r['task']==task]]])
    section('### 8.4 正式ZeRO-2续训与近期更新独立观察')
    range_table([('1–40',1,40),('41–320',41,320),('321–截止',321,updates)])
    table(['范围','工具','执行动作','H命中','H/执行','动态GT'],[[lab,t,sc(rr,t,'executed'),sc(rr,t,'hit'),ratio(sc(rr,t,'hit'),sc(rr,t,'executed')),sc(rr,t,'gt')] for lab,rr in [('全量保留',retained),('最近100保留',[r for r in retained if r['optimizer_step']>updates-100])] for t in tools])
    add('上述表同时包含数据构成变化、BGE修复及长度筛选效应；没有同样本同预算对照，不能把H的前后差异归因为能力提升。')
    section('### 8.5 提前停止后的最终600条测试集评估')
    render_tests(ctx,add,table,section,pct,ratio)
    section('## 9. NC完整输入和LLM原始输出','examples')
    add('案例沿用3_5分类：最早完整、格式终止、非法后继续、B满分但未恢复、候选内错误、H改写、末段完整；另保留长度丢弃案例。按类别确定性抽样，不作为代表性成功率。每轮完整输入由实际token前缀还原，不手工补写analysis。')
    labels=['最早完整NC','格式错误终止','非法action后继续','B满分但图未完整恢复','候选内错误得0.1','改写获得H','末段完整NC','长度丢弃']
    cases=[(label,ctx['example_raw'][label]) for label in labels if label in ctx['example_raw']]
    table(['案例','用途','source_index','row.step/窗口','实际更新','S/B/T/H→R','丢弃'],[[f'[案例{i}](#case-{i})',label,e['summary']['source'],f'{e["summary"]["step"]}/{e["summary"]["window"]}',e['summary']['optimizer_step'],'/'.join(f'{e["summary"][k]:.2f}' for k in ['S','B','T','H'])+f'→{e["summary"]["R"]:.2f}',e['summary']['discarded']] for i,(label,e) in enumerate(cases,1)],False)
    section('## 10. 完整NC候选逐轮实录','transcripts')
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file('/home/temptest/models/Qwen2.5-7B-Instruct/tokenizer.json')
    exported=[]
    for i,(label,e) in enumerate(cases,1):
        a,r,segs=e['row'],e['summary'],e['segments']
        section(f'### 10.{i} 案例{i}：{label} / source_index={r["source"]} / step={r["step"]}',f'case-{i}')
        add(f'输入{"dirty" if r["dirty"] else "clean"}；格式错误={a["format_errors"]}；执行错误={a["errors"]}；完整图exact={r["full"]}。原始运行{r["run"]}，字节偏移{r["offset"]}。')
        table(['同源候选','有效派发','执行非法','S','B','T','H','R','advantage','丢弃'],[[j,sum(q['stages'][t]['dispatch'] for t in tools),q['illegal']]+[q[k] for k in ['S','B','T','H','R','advantage','discarded']] for j,q in enumerate(groups[r['step'],r['source']],1)],False)
        if not r['token_available']:
            add('该候选因长度筛选替换了token字段，不能伪称恢复原始token；以下是实际保存的prompt/messages。')
            add('```json\n'+json.dumps(dict(prompt=a['prompt'],messages=a['messages']),ensure_ascii=False,indent=2)+'\n```')
            exported.append(dict(label=label,summary=r,raw_tokens_available=False,prompt=a['prompt'],messages=a['messages']));continue
        prompt=[v for v,m in zip(a['prompt_ids'],a['prompt_mask']) if m]
        c,cm,tm=a['completion_ids'],a['completion_mask'],a['tool_mask']
        starts=[j for j in range(len(c)) if cm[j] and tm[j] and (j==0 or not(cm[j-1] and tm[j-1]))]
        inputs=[];outputs=[];ai=0
        for msg in a['messages']:
            if msg['role']=='assistant':
                effective=tok.decode(prompt+[c[j] for j in range(starts[ai]) if cm[j]],skip_special_tokens=False)
                output=tok.decode(segs[ai],skip_special_tokens=False);inputs.append(effective);outputs.append(output)
                section(f'#### 第{ai+1}次LLM生成：应处阶段{stages[ai]}')
                add(f'有效输入{len(prompt)+sum(cm[:starts[ai]])} token；模型输出{len(segs[ai])} token。')
                add('<details>\n<summary>展开本轮完整LLM输入prompt</summary>\n\n```text\n'+effective+'\n```\n\n</details>')
                add('**LLM原始输出**\n\n```text\n'+output+'\n```');ai+=1
            elif msg['role']=='tool':add('**外部Python反馈**\n\n```text\n'+msg['content']+'\n```')
        exported.append(dict(label=label,summary=r,effective_inputs_decoded=inputs,generated_segments_decoded=outputs,messages=a['messages']))
    (out/'extracted_nc_inputs_outputs.json').write_text(json.dumps(exported,ensure_ascii=False,indent=2))
    section('## 11. 证据、检查与解释边界','evidence')
    for name,title in [('statistics.json','统计与来源哈希'),('trajectory_summaries.jsonl','逐候选摘要'),('error_category_mapping.json','错误分类原文映射'),('extracted_nc_inputs_outputs.json','案例原文'),('progress_snapshot.json','固定进度'),('table_validation.json','表格校验'),('analyze_3_6_report.py','提取脚本'),('render_3_6_report.py','报告生成脚本')]:
        add(f'- [{title}](analysis/{out.name}/{name})')
    add('- 验证每窗口12候选、每源六候选、窗口连续；R=S+B+T+H；格式错T=0；拒绝前后图哈希一致；保留数量与performance一致。')
    add('- 验证表格显示宽度逐行一致，中文使用East Asian Width；动态原始报错不作为列名。分母、计数与范围在对应表旁说明。')
    add('- 本次保留旧报告备份；基座测试按用户要求跳过，旧实验基座结果不复制成3_6结果。仅CPU分析，不干扰测试进程。')
    summary=dict(time=ctx['now'].isoformat(),cutoff_window=cutoff,engine_global_step=updates,all_candidates=len(rows),kept_candidates=len(retained),discarded_candidates=len(rows)-len(retained),groups=len(groups),reward_means={k:m(rows,k) for k in ['S','B','T','H','R']},accuracy=m(rows,'correct'),dirty_full=sum(r['full'] for r in dirty),dirty_candidates=len(dirty),manifest=ctx['manifest'],anomalies=dict(ctx['anomalies']),train_data_sha256=hashlib.sha256(ctx['data_path'].read_bytes()).hexdigest(),progress_snapshot=ctx['progress'],bin_axis='actual_optimizer_update',tables=len(table_audits))
    (out/'statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'progress_snapshot.json').write_text(json.dumps(ctx['progress'],indent=2))
    (out/'table_validation.json').write_text(json.dumps(table_audits,indent=2))
    for name in ['analyze_3_6_report.py','render_3_6_report.py']:shutil.copy2(Path(__file__).parent/name,out/name)
    target=Path(os.environ.get('TASK_REPORT_TARGET',str(B/'reports/3_6.md')))
    if target.exists():shutil.copy2(target,out/'previous_3_6.md')
    report='\n'.join(parts)
    (out/'3_6.md').write_text(report)
    tmp=target.with_suffix('.md.tmp');tmp.write_text(report);os.replace(tmp,target)
    print(json.dumps({k:v for k,v in summary.items() if k!='manifest'},ensure_ascii=False,indent=2),flush=True)


def render_tests(ctx,add,table,section,pct,ratio):
    current=ctx['current'];pointer=current/'final_evaluation_pointer.json'
    if not pointer.exists():add('尚无独立策略测试记录。');return
    folder=Path(json.loads(pointer.read_text())['path'])
    manifest=json.loads((folder/'manifest.json').read_text())
    add(f'模型使用checkpoint-{manifest["checkpoint_step"]}，由原始Qwen基座加对应LoRA合并；基座测试跳过。clean-policy/noisy-policy各600条，greedy，每轮512、completion8704，不对测试集应用训练5120丢弃规则。')
    pred={}
    for cond in ['restored','clean_policy']:
        p=folder/cond/'summary.json'
        if not p.exists():add(f'{cond}未完成，结果不预填。');return
        pred[cond]=[json.loads(l) for l in (folder/cond/'predictions.jsonl').open()]
        assert len(pred[cond])==600 and len({r['source_index'] for r in pred[cond]})==600
    data={s['source_index']:s for s in json.loads((ctx['B'].parents[1]/'data/3_5_1_3_6/test_episodes.json').read_text())}
    derived={};operation={}
    for cond,rr in pred.items():
        dd=[];ops=defaultdict(Counter)
        for r in rr:
            s=data[r['source_index']];clean=s['clean_graph'];cn=ctx['nodes'](clean);ce=ctx['edges'](clean);fg=r['final_graph'];fn=ctx['nodes'](fg);fe=ctx['edges'](fg)
            dirty=(ctx['nodes'](s['noisy_graph'])!=cn or ctx['edges'](s['noisy_graph'])!=ce) if cond=='restored' else False
            dd.append(dict(source=r['source_index'],task=r['task_type'],correct=r['metrics']['prediction_correct'],format=r['metrics']['format_valid'],illegal=any(e.get('error') for e in r['events']),edit=bool(r['executed_actions']),full=fn==cn and fe==ce,structure=set(fn)==set(cn) and fe==ce,dirty=dirty,missing_nodes=len(set(cn)-set(fn)),extra_nodes=len(set(fn)-set(cn)),missing_edges=len(ce-fe),extra_edges=len(fe-ce)))
            for e in r['events']:
                t=e['expected_tool'];acts=(e.get('call') or {}).get('actions',[])
                ops[t]['dispatch']+=bool(e.get('format_valid'));ops[t]['illegal']+=bool(e.get('error'))
                if e.get('error') or not e.get('format_valid'):continue
                ops[t]['empty']+=not acts;ops[t]['executed']+=len(acts)
                for a in acts:
                    hit=(a['node'] not in cn) if t=='del_node' else (tuple(sorted((a['u'],a['v']))) not in ce) if t=='del_edge' else (tuple(sorted((a['u'],a['v']))) in ce) if t=='add_edge' else a['node'] in cn and a['text']==cn[a['node']]
                    ops[t]['reference']+=hit
        derived[cond]=dd;operation[cond]=ops
    section('#### 8.5.1 总体、NC和LC正确率')
    table(['条件','总体','NC','LC'],[[cond]+[ratio(sum(r['correct'] for r in dd if task is None or r['task']==task),sum(task is None or r['task']==task for r in dd)) for task in [None,'NC','LC']] for cond,dd in derived.items()],False)
    section('#### 8.5.2 格式、执行、编辑和最终图质量')
    table(['条件','格式完整','图非法轨迹','发生编辑','完整图exact','结构exact','dirty完整恢复'],[[cond]+[ratio(sum(r[k] for r in dd),len(dd)) for k in ['format','illegal','edit','full','structure']]+[ratio(sum(r['dirty'] and r['full'] for r in dd),sum(r['dirty'] for r in dd))] for cond,dd in derived.items()],False)
    table(['条件','正常节点缺失','额外节点','缺失边','额外边'],[[cond]+[sum(r[k] for r in dd) for k in ['missing_nodes','extra_nodes','missing_edges','extra_edges']] for cond,dd in derived.items()],False)
    section('#### 8.5.3 四工具行为与clean参考匹配')
    table(['条件','工具','有效派发','合法空','非法阶段','已执行动作','clean匹配','匹配比例'],[[cond,t,o['dispatch'],o['empty'],o['illegal'],o['executed'],o['reference'],ratio(o['reference'],o['executed'])] for cond,ops in operation.items() for t,o in ops.items()])
    section('#### 8.5.4 奖励分项与解释边界')
    table(['条件','S','B','T','H','R'],[[cond]+[sum(r['metrics'].get(k,0) for r in rr)/len(rr) for k in ['structure_reward','behavior_reward','task_reward','hit_reward','total']] for cond,rr in pred.items()],False)
    add('测试R只作机制解释；clean-policy仍运行四工具，误编辑率需结合图exact判断。无新增基座测试，不计算伪造的相对基座提升。')
    section('#### 8.5.5 同一测试样本在noisy与clean条件下的配对变化')
    a={r['source']:r for r in derived['restored']};b={r['source']:r for r in derived['clean_policy']}
    assert a.keys()==b.keys()
    table(['noisy正确','clean正确','样本数'],[[x,y,sum(bool(a[k]['correct'])==x and bool(b[k]['correct'])==y for k in a)] for x,y in [(True,True),(True,False),(False,True),(False,False)]],False)
    add('这是同一策略两种输入条件的配对，不是基座→策略的配对。')
    section('#### 8.5.6 本次提前停止实验结论')
    for cond,dd in derived.items():add(f'{cond}：正确率{ratio(sum(r["correct"] for r in dd),len(dd))}；完整图exact{ratio(sum(r["full"] for r in dd),len(dd))}。最终预测与去噪分别评价，训练与测试不混算。')
    add(f'完整测试证据：`{folder}`；manifest含adapter与测试集哈希，predictions.jsonl保存每轮输入、输出、事件、最终图及评分。')
    (ctx['out']/'final_test_statistics.json').write_text(json.dumps(dict(manifest=manifest,derived=derived,operations=operation),ensure_ascii=False,indent=2))
