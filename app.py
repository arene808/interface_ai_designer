# -*- coding: utf-8 -*-
"""
精益求径 — Flask Web API 服务器
==============================
REST API: AI预测 + 数据管理 + 模型重训练
"""

import sys, os, json, io
import numpy as np

impl_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, impl_dir)

from flask import Flask, request, jsonify, render_template, send_file
from ai_model import get_model, InterfaceAIModel
from data_manager import (
    init_db, seed_from_json, import_csv, export_csv_template,
    get_all_groups, get_group_results, get_training_data,
    get_statistics, export_full_json, get_all_fourier_features
)

app = Flask(__name__)

# 服务启动时初始化
model = None

def init_model():
    global model
    if model is None:
        print("[INFO] 初始化数据库...")
        init_db()
        n = seed_from_json()
        print(f"[INFO] 数据库就绪 ({n} 组)")

        # 检查是否有真实实验数据
        X_real, y_real, groups = get_training_data()
        model = get_model()  # 先获取模型实例

        if X_real is not None and len(X_real) >= 3:
            print(f"[INFO] 发现 {len(X_real)} 组真实实验数据，使用真实数据训练")
            result = model.train_from_real_data(X_real, y_real)
            print(f"[INFO] 真实数据训练完成: R2={result.get('train_r2', 'N/A')}")
        else:
            print(f"[INFO] 真实数据不足 ({len(groups) if X_real is not None else 0} 组)，使用合成数据预训练")
            model.train(n_synthetic=300)

        print(f"[INFO] 模型就绪 - 强度范围: {model.strength_min:.1f}~{model.strength_max:.1f} MPa")
    return model


# ============================================================
# 页面
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


# ============================================================
# AI 预测 API
# ============================================================
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'message': '精益求径 AI服务运行中'})


@app.route('/api/model_info')
def model_info():
    m = init_model()
    stats = get_statistics()
    return jsonify({
        'status': 'trained',
        'strength_range': m.get_strength_range(),
        'n_reference_groups': len(m.reference_groups),
        'reference_names': list(m.reference_groups.keys()),
        'database_stats': stats,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    m = init_model()
    data = request.get_json()
    if not data or 'target_strength' not in data:
        return jsonify({'error': 'Missing required field: target_strength'}), 400

    target_strength = float(data['target_strength'])
    preferred_type = data.get('preferred_type', None)

    lo, hi = m.get_strength_range()
    if target_strength < lo * 0.5 or target_strength > hi * 1.5:
        return jsonify({
            'error': f'Strength out of range ({lo:.1f}~{hi:.1f} MPa)',
            'valid_range': [lo, hi],
        }), 400

    try:
        result = m.inverse_design(target_strength, preferred_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reference_curves')
def reference_curves():
    m = init_model()
    curves = m.get_reference_curves()
    return jsonify({'curves': curves, 'count': len(curves)})


@app.route('/api/curve_types')
def curve_types():
    return jsonify({'types': [
        {'id': 'flat', 'name': '平面界面', 'name_en': 'Flat',
         'description': '纯粘附界面，无机械互锁。强度最低，制造最简单。',
         'strength_range': '1.0~1.6 MPa', 'color': '#9E9E9E'},
        {'id': 'sinusoidal', 'name': '正弦界面', 'name_en': 'Sinusoidal',
         'description': '连续曲率波状界面，无尖角应力集中。中等互锁强度。',
         'strength_range': '1.4~2.6 MPa', 'color': '#2196F3'},
        {'id': 'sawtooth', 'name': '锯齿界面', 'name_en': 'Sawtooth',
         'description': '对称三角波界面，尖角增强方向性互锁。中高强度。',
         'strength_range': '1.6~2.5 MPa', 'color': '#FF5722'},
        {'id': 'dovetail', 'name': '燕尾界面', 'name_en': 'Dovetail',
         'description': '梯形机械锁扣界面，交替方向锁死。最高互锁强度。',
         'strength_range': '2.0~3.5 MPa', 'color': '#4CAF50'},
    ]})


# ============================================================
# 数据管理 API
# ============================================================
@app.route('/api/data/stats')
def data_stats():
    stats = get_statistics()
    m = init_model()
    stats['strength_range'] = m.get_strength_range()
    stats['model_source'] = 'real_data' if get_training_data()[0] is not None and len(get_training_data()[0]) >= 3 else 'synthetic'
    return jsonify(stats)


@app.route('/api/data/groups')
def data_groups():
    groups = get_all_groups()
    # 附加测试统计
    for g in groups:
        results = get_group_results(g['group_name'])
        debonds = [r['sigma_debond'] for r in results if r['sigma_debond'] is not None and r['sigma_debond'] > 0]
        g['n_tests'] = len(results)
        g['n_valid'] = len(debonds)
        g['avg_sigma_debond'] = float(np.mean(debonds)) if debonds else None
        g['std_sigma_debond'] = float(np.std(debonds)) if len(debonds) > 1 else None
    return jsonify({'groups': groups, 'count': len(groups)})


@app.route('/api/data/groups/<group_name>')
def data_group_detail(group_name):
    results = get_group_results(group_name)
    return jsonify({'group_name': group_name, 'results': results, 'count': len(results)})


@app.route('/api/data/import', methods=['POST'])
def data_import():
    """导入CSV实验数据"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # 保存临时文件
    tmp_path = os.path.join(impl_dir, '_temp_import.csv')
    file.save(tmp_path)

    try:
        result = import_csv(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/api/data/template')
def data_template():
    """下载CSV导入模板"""
    path = export_csv_template()
    return send_file(path, as_attachment=True, download_name='实验数据导入模板.csv')


@app.route('/api/data/export')
def data_export():
    """导出完整数据JSON"""
    path = export_full_json()
    return send_file(path, as_attachment=True, download_name='实验数据_导出.json')


@app.route('/api/data/add_result', methods=['POST'])
def data_add_result():
    """手动添加单条测试结果"""
    data = request.get_json()
    if not data or 'group_name' not in data:
        return jsonify({'error': 'Missing group_name'}), 400

    import sqlite3
    from data_manager import get_conn

    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO test_results
                (group_name, specimen_id, sigma_debond, sigma_ultimate, failure_mode, test_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['group_name'],
            data.get('specimen_id', ''),
            data.get('sigma_debond'),
            data.get('sigma_ultimate'),
            data.get('failure_mode', ''),
            data.get('test_date', ''),
            data.get('notes', '手动录入'),
        ))
        conn.commit()
        new_id = c.lastrowid
        return jsonify({'status': 'ok', 'id': new_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


# ============================================================
# 模型重训练 API
# ============================================================
@app.route('/api/retrain', methods=['POST'])
def retrain():
    """使用数据库中的真实数据重新训练模型"""
    global model

    X_real, y_real, groups = get_training_data()

    if X_real is None or len(X_real) < 3:
        return jsonify({
            'status': 'error',
            'message': f'真实数据不足: {len(groups)} 组有实验数据，至少需要3组。',
            'groups_with_data': groups,
        }), 400

    m = get_model()
    result = m.train_from_real_data(X_real, y_real)
    model = m

    result['groups_used'] = groups
    result['n_groups'] = len(groups)

    return jsonify(result)


@app.route('/api/retrain_synthetic', methods=['POST'])
def retrain_synthetic():
    """回退到合成数据训练"""
    global model
    m = get_model()
    data = request.get_json() or {}
    n_samples = data.get('n_samples', 300)
    result = m.train(n_synthetic=n_samples)
    model = m
    return jsonify(result)


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  精益求径 — AI界面逆设计Web服务")
    print("  多材料3D打印界面互锁参数预测引擎")
    print("=" * 60)
    print()
    print("  访问地址: http://127.0.0.1:5000")
    print()
    print("=" * 60)

    init_model()

    app.run(host='0.0.0.0', port=5000, debug=False)