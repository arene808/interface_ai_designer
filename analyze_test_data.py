# -*- coding: utf-8 -*-
'''
分析 TPU-PLA 拉伸测试真实数据
- 解析所有 .xls 文件提取应力-应变曲线
- 自动检测脱粘强度 (sigma_debond) 和极限强度 (sigma_ultimate)
- 绘制分析图表
- 生成 AI 训练数据库
'''
import os, sys, json, re, warnings
import numpy as np
import xlrd

warnings.filterwarnings('ignore')

DATA_DIR = r'E:\Backup_WK\资料\精益求“径”\tpu-pla20260627'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_output')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 解析所有 .xls 文件
# ============================================================
def parse_xls(filepath):
    '''解析拉伸测试 .xls 文件，返回应力-应变数据'''
    wb = xlrd.open_workbook(filepath)
    sh = wb.sheet_by_index(0)
    
    data = {'time': [], 'force': [], 'disp': [], 'strain': [], 'stress': []}
    for i in range(1, sh.nrows):  # skip header
        try:
            data['time'].append(float(sh.cell_value(i, 0)))
            data['force'].append(float(sh.cell_value(i, 1)))
            data['disp'].append(float(sh.cell_value(i, 2)))
            data['strain'].append(float(sh.cell_value(i, 6)))
            data['stress'].append(float(sh.cell_value(i, 7)))
        except (ValueError, IndexError):
            continue
    
    for k in data:
        data[k] = np.array(data[k])
    return data

def detect_debond_stress(stress, strain, min_stress_threshold=2.0):
    '''
    自动检测脱粘应力 - 第一个显著应力下降点
    使用滑动窗口检测应力突变
    
    返回: (sigma_debond, debond_index, failure_mode_guess)
    '''
    n = len(stress)
    if n < 100:
        return None, None, 'unknown'
    
    # 1. 找到峰值应力
    peak_idx = np.argmax(stress)
    peak_stress = stress[peak_idx]
    
    if peak_stress < min_stress_threshold:
        return peak_stress, peak_idx, 'low_strength'
    
    # 2. 从峰值后找第一个显著下降（>10% drop within 20 data points）
    window = 20
    drop_threshold = 0.10  # 10% drop
    
    for i in range(peak_idx, min(n - window, n - 1)):
        current = stress[i]
        future_min = np.min(stress[i+1:i+window+1])
        drop_ratio = (current - future_min) / max(current, 0.01)
        
        if drop_ratio > drop_threshold:
            debond_idx = i
            sigma_debond = stress[debond_idx]
            
            # 判断断裂模式
            if sigma_debond > 0.85 * peak_stress:
                mode = 'CF-T'  # 内聚断裂-转移 (cohesive failure - transfer)
            elif sigma_debond > 0.5 * peak_stress:
                mode = 'CF-M'  # 内聚断裂-混合
            else:
                mode = 'AF'    # 界面脱粘 (adhesive failure)
            
            return float(sigma_debond), int(debond_idx), mode
    
    # 无显著下降 -> 整体脱粘
    return float(peak_stress), int(peak_idx), 'AF'

def analyze_specimen(filepath):
    '''分析单个试样'''
    data = parse_xls(filepath)
    stress = data['stress']
    strain = data['strain']
    
    sigma_ultimate = float(np.max(stress))
    sigma_debond, debond_idx, mode_guess = detect_debond_stress(stress, strain)
    
    # 如果没有检测到脱粘，使用峰值
    if sigma_debond is None:
        sigma_debond = sigma_ultimate
        debond_idx = np.argmax(stress)
    
    # 计算模量 (0.02% ~ 0.2% 应变区间的斜率)
    mask = (strain >= 0.02) & (strain <= 0.2)
    if np.sum(mask) > 5:
        E = float(np.polyfit(strain[mask], stress[mask], 1)[0])
    else:
        E = None
    
    # 韧性 (应力-应变曲线下面积)
    toughness = float(np.trapz(stress, strain))
    
    # 断裂应变 (应力降到峰值20%时的应变)
    peak_idx = np.argmax(stress)
    post_peak = stress[peak_idx:]
    below_20 = np.where(post_peak < 0.2 * sigma_ultimate)[0]
    if len(below_20) > 0:
        failure_strain = float(strain[peak_idx + below_20[0]])
    else:
        failure_strain = float(strain[-1])
    
    return {
        'stress': stress.tolist(),
        'strain': strain.tolist(),
        'sigma_debond': sigma_debond,
        'sigma_ultimate': sigma_ultimate,
        'debond_index': debond_idx,
        'failure_mode': mode_guess,
        'elastic_modulus': E,
        'toughness': toughness,
        'failure_strain': failure_strain,
        'data_points': len(stress),
    }

print('=' * 60)
print('解析拉伸测试数据...')
print('=' * 60)

# 收集所有文件
groups = {}
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.xls'):
        continue
    # 跳过临时文件
    if fname.startswith('~'):
        continue
    
    match = re.match(r'([A-Z]\d*)-(\d+|P\d+)\.xls', fname)
    if not match:
        print(f'  [SKIP] Cannot parse filename: {fname}')
        continue
    
    group = match.group(1)
    specimen = match.group(2)
    
    if group not in groups:
        groups[group] = []
    
    fpath = os.path.join(DATA_DIR, fname)
    print(f'  Analyzing {fname}...', end=' ')
    try:
        result = analyze_specimen(fpath)
        result['filename'] = fname
        result['group'] = group
        result['specimen_id'] = specimen
        groups[group].append(result)
        print(f'OK  sigma_d={result["sigma_debond"]:.2f}  sigma_u={result["sigma_ultimate"]:.2f}  mode={result["failure_mode"]}')
    except Exception as e:
        print(f'FAIL: {e}')

# 打印汇总
print()
print('=' * 60)
print('数据汇总')
print('=' * 60)
for group in sorted(groups.keys()):
    specimens = groups[group]
    debonds = [s['sigma_debond'] for s in specimens]
    ultimates = [s['sigma_ultimate'] for s in specimens]
    modes = [s['failure_mode'] for s in specimens]
    
    avg_d = np.mean(debonds)
    std_d = np.std(debonds, ddof=1) if len(debonds) > 1 else 0
    avg_u = np.mean(ultimates)
    std_u = np.std(ultimates, ddof=1) if len(ultimates) > 1 else 0
    
    mode_counts = {m: modes.count(m) for m in set(modes)}
    
    print(f'  {group} ({len(specimens)} specimens):')
    print(f'    sigma_debond  = {avg_d:.2f} +/- {std_d:.2f} MPa')
    print(f'    sigma_ultimate = {avg_u:.2f} +/- {std_u:.2f} MPa')
    print(f'    failure_modes  = {mode_counts}')

# 保存到JSON
print()
summary = {}
for group in sorted(groups.keys()):
    specimens = groups[group]
    debonds = [s['sigma_debond'] for s in specimens]
    ultimates = [s['sigma_ultimate'] for s in specimens]
    modes = [s['failure_mode'] for s in specimens]
    
    summary[group] = {
        'n_specimens': len(specimens),
        'sigma_debond_mean': float(np.mean(debonds)),
        'sigma_debond_std': float(np.std(debonds, ddof=1)) if len(debonds) > 1 else 0,
        'sigma_debond_values': [float(d) for d in debonds],
        'sigma_ultimate_mean': float(np.mean(ultimates)),
        'sigma_ultimate_std': float(np.std(ultimates, ddof=1)) if len(ultimates) > 1 else 0,
        'sigma_ultimate_values': [float(u) for u in ultimates],
        'failure_modes': modes,
        'failure_mode_dominant': max(set(modes), key=modes.count),
        'specimens': [{
            'id': s['specimen_id'],
            'filename': s['filename'],
            'sigma_debond': s['sigma_debond'],
            'sigma_ultimate': s['sigma_ultimate'],
            'failure_mode': s['failure_mode'],
            'elastic_modulus': s['elastic_modulus'],
            'toughness': s['toughness'],
            'failure_strain': s['failure_strain'],
        } for s in specimens],
    }

with open(os.path.join(OUT_DIR, 'mechanical_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f'Summary saved to {OUT_DIR}/mechanical_summary.json')

# 保存完整应力-应变数据 (压缩版, 每10个点取1个)
full_data = {}
for group in sorted(groups.keys()):
    full_data[group] = []
    for s in groups[group]:
        # 降采样以减小文件
        step = max(1, len(s['stress']) // 500)
        full_data[group].append({
            'id': s['specimen_id'],
            'stress': s['stress'][::step],
            'strain': s['strain'][::step],
            'sigma_debond': s['sigma_debond'],
            'sigma_ultimate': s['sigma_ultimate'],
            'failure_mode': s['failure_mode'],
        })

with open(os.path.join(OUT_DIR, 'stress_strain_curves.json'), 'w', encoding='utf-8') as f:
    json.dump(full_data, f, ensure_ascii=False)
print(f'Curves saved to {OUT_DIR}/stress_strain_curves.json')

print()
print('Data analysis complete!')