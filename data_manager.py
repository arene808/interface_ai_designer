# -*- coding: utf-8 -*-
"""
精益求径 — 实验数据管理器
=========================
SQLite 数据库 + CSV/JSON 导入 + 数据查询

数据库表:
  - experiment_groups: 实验组元数据（组名、界面类型、参数）
  - test_results:      拉伸测试结果（σ_debond, σ_ultimate, failure_mode）
  - curve_data:        界面曲线坐标点
"""

import sqlite3
import json
import csv
import os
import numpy as np
from datetime import datetime

DB_PATH = None  # 在init时设置

# ============================================================
# 数据库初始化
# ============================================================
def init_db(db_path=None):
    global DB_PATH
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_data.db')
    DB_PATH = db_path

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 实验组表
    c.execute('''
        CREATE TABLE IF NOT EXISTS experiment_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT UNIQUE NOT NULL,
            interface_type TEXT NOT NULL,
            amplitude REAL,
            period REAL DEFAULT 4.8,
            neck_width REAL,
            head_width REAL,
            dovetail_depth REAL,
            sawtooth_asymmetry REAL DEFAULT 0.5,
            status TEXT DEFAULT 'designed',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    ''')

    # 测试结果表
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT NOT NULL,
            specimen_id TEXT,
            sigma_debond REAL,
            sigma_ultimate REAL,
            failure_mode TEXT,
            test_date TEXT,
            operator TEXT,
            temperature REAL,
            strain_rate REAL,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (group_name) REFERENCES experiment_groups(group_name)
        )
    ''')

    # 傅里叶特征表
    c.execute('''
        CREATE TABLE IF NOT EXISTS fourier_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT UNIQUE NOT NULL,
            A1 REAL, A2 REAL, A3 REAL, A4 REAL, A5 REAL,
            rms_roughness REAL, curvature_max REAL, tortuosity REAL,
            total_power REAL,
            spectral_decay_exponent REAL,
            FOREIGN KEY (group_name) REFERENCES experiment_groups(group_name)
        )
    ''')

    conn.commit()
    conn.close()
    return DB_PATH


def get_conn():
    if DB_PATH is None:
        init_db()
    return sqlite3.connect(DB_PATH)


# ============================================================
# 从现有 JSON 初始化数据库
# ============================================================
def seed_from_json(json_path=None):
    """从 interface_fourier_features.json 导入初始数据"""
    if json_path is None:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'interface_fourier_features.json')

    if not os.path.exists(json_path):
        print(f"[WARN] JSON not found: {json_path}")
        return 0

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conn = get_conn()
    c = conn.cursor()
    count = 0

    for entry in data:
        name = entry['group']
        itype = entry['interface_type']
        params = entry.get('params', {})
        fourier = entry.get('fourier', {})
        geo = entry.get('geometric_features', {})
        exp = entry.get('experiment', {})

        # 插入实验组
        c.execute('''
            INSERT OR IGNORE INTO experiment_groups
                (group_name, interface_type, amplitude, period, neck_width, head_width,
                 dovetail_depth, sawtooth_asymmetry, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'designed')
        ''', (
            name, itype,
            params.get('amplitude'), params.get('period', 4.8),
            params.get('neck_width'), params.get('head_width'),
            params.get('dovetail_depth'), 0.5
        ))

        # 插入傅里叶特征
        amps = fourier.get('amplitudes', [0]*10)
        c.execute('''
            INSERT OR REPLACE INTO fourier_features
                (group_name, A1, A2, A3, A4, A5, rms_roughness, curvature_max,
                 tortuosity, total_power, spectral_decay_exponent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name,
            amps[0] if len(amps) > 0 else 0,
            amps[1] if len(amps) > 1 else 0,
            amps[2] if len(amps) > 2 else 0,
            amps[3] if len(amps) > 3 else 0,
            amps[4] if len(amps) > 4 else 0,
            geo.get('rms_roughness', 0),
            geo.get('curvature_max', 0),
            geo.get('tortuosity', 1.0),
            fourier.get('total_power', 0),
            fourier.get('spectral_decay_exponent'),
        ))

        # 插入实验数据（如果有）
        if exp.get('sigma_debond') is not None or exp.get('sigma_ultimate') is not None:
            c.execute('''
                INSERT INTO test_results
                    (group_name, sigma_debond, sigma_ultimate, failure_mode, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                name, exp.get('sigma_debond'), exp.get('sigma_ultimate'),
                exp.get('failure_mode'), 'Imported from JSON'
            ))

        count += 1

    conn.commit()
    conn.close()
    print(f"[OK] Seeded {count} groups from JSON")
    return count


# ============================================================
# CSV 导入
# ============================================================
def import_csv(csv_path):
    """导入CSV格式的实验数据

    CSV 列要求:
      group_name, sigma_debond, sigma_ultimate, failure_mode, specimen_id(可选), test_date(可选)

    示例:
      N2,8.5,12.3,CF-T,specimen-1,2026-06-20
      N2,7.9,11.8,AF,specimen-2,2026-06-20
    """
    conn = get_conn()
    c = conn.cursor()
    imported = 0
    errors = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                group = row.get('group_name', '').strip()
                if not group:
                    errors.append(f"Row {i+2}: missing group_name")
                    continue

                sigma_d = float(row.get('sigma_debond', 0) or 0)
                sigma_u = float(row.get('sigma_ultimate', 0) or 0)
                failure = row.get('failure_mode', '').strip() or None
                specimen = row.get('specimen_id', '').strip() or f'{group}-{imported+1:03d}'
                test_date = row.get('test_date', '').strip() or datetime.now().strftime('%Y-%m-%d')

                c.execute('''
                    INSERT INTO test_results
                        (group_name, specimen_id, sigma_debond, sigma_ultimate,
                         failure_mode, test_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (group, specimen, sigma_d, sigma_u, failure, test_date))
                imported += 1
            except Exception as e:
                errors.append(f"Row {i+2}: {e}")

    conn.commit()
    conn.close()

    result = {'imported': imported, 'errors': errors}
    print(f"[OK] CSV imported: {imported} rows")
    if errors:
        for e in errors[:5]:
            print(f"  [ERR] {e}")
    return result


def export_csv_template():
    """导出CSV模板（含已有实验组名）"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT group_name, interface_type, amplitude FROM experiment_groups")
    groups = c.fetchall()
    conn.close()

    template = "group_name,specimen_id,sigma_debond,sigma_ultimate,failure_mode,test_date,notes\n"
    for g in groups:
        template += f"{g[0]},, , , , ,{g[1]} A={g[2]}mm\n"

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'import_template.csv')
    with open(template_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(template)
    return template_path


# ============================================================
# 查询接口
# ============================================================
def get_all_groups():
    """获取所有实验组"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM experiment_groups ORDER BY group_name")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def get_group_results(group_name):
    """获取某个实验组的所有测试结果"""
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM test_results
        WHERE group_name = ?
        ORDER BY created_at DESC
    ''', (group_name,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def get_training_data():
    """获取可用于AI训练的数据：
    返回: (X: 8D特征矩阵, y: 平均脱粘强度, groups: 组名列表)
    """
    conn = get_conn()
    c = conn.cursor()

    # 获取所有有测试数据的组
    c.execute('''
        SELECT DISTINCT tr.group_name
        FROM test_results tr
        WHERE tr.sigma_debond IS NOT NULL AND tr.sigma_debond > 0
    ''')
    tested_groups = [r[0] for r in c.fetchall()]

    if not tested_groups:
        conn.close()
        return None, None, []

    X_list, y_list, groups = [], [], []

    for group in tested_groups:
        # 获取傅里叶特征
        c.execute('''
            SELECT A1, A2, A3, A4, A5, rms_roughness, curvature_max, tortuosity
            FROM fourier_features WHERE group_name = ?
        ''', (group,))
        feat = c.fetchone()
        if feat is None:
            continue

        # 获取平均脱粘强度
        c.execute('''
            SELECT AVG(sigma_debond), COUNT(*)
            FROM test_results
            WHERE group_name = ? AND sigma_debond IS NOT NULL AND sigma_debond > 0
        ''', (group,))
        avg_s, n = c.fetchone()
        if avg_s is None or n == 0:
            continue

        X_list.append(list(feat))
        y_list.append(avg_s)
        groups.append(group)

    conn.close()

    if not X_list:
        return None, None, []

    return np.array(X_list), np.array(y_list), groups


def get_all_fourier_features():
    """获取所有组的傅里叶特征"""
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        SELECT ff.*, eg.interface_type, eg.amplitude
        FROM fourier_features ff
        JOIN experiment_groups eg ON ff.group_name = eg.group_name
        ORDER BY ff.group_name
    ''')
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def get_statistics():
    """获取数据库统计信息"""
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM experiment_groups")
    n_groups = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM test_results")
    n_tests = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT group_name) FROM test_results WHERE sigma_debond IS NOT NULL AND sigma_debond > 0")
    n_tested = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM fourier_features")
    n_features = c.fetchone()[0]

    conn.close()
    return {
        'n_groups': n_groups,
        'n_tests': n_tests,
        'n_tested_groups': n_tested,
        'n_fourier_features': n_features,
        'can_train': n_tested >= 3,
    }


def export_full_json(output_path=None):
    """导出完整数据为JSON（含实验数据）"""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_data_export.json')

    conn = get_conn()
    c = conn.cursor()

    c.execute('''
        SELECT eg.*, ff.A1, ff.A2, ff.A3, ff.A4, ff.A5,
               ff.rms_roughness, ff.curvature_max, ff.tortuosity,
               ff.total_power, ff.spectral_decay_exponent
        FROM experiment_groups eg
        LEFT JOIN fourier_features ff ON eg.group_name = ff.group_name
        ORDER BY eg.group_name
    ''')
    groups = c.fetchall()
    cols = [d[0] for d in c.description]

    result = []
    for g in groups:
        gdict = dict(zip(cols, g))
        c.execute('''
            SELECT sigma_debond, sigma_ultimate, failure_mode, specimen_id, test_date
            FROM test_results WHERE group_name = ?
        ''', (gdict['group_name'],))
        tests = c.fetchall()
        gdict['tests'] = [
            {'sigma_debond': t[0], 'sigma_ultimate': t[1],
             'failure_mode': t[2], 'specimen_id': t[3], 'test_date': t[4]}
            for t in tests
        ]
        # 计算平均值
        if tests:
            debonds = [t[0] for t in tests if t[0] is not None and t[0] > 0]
            ultimates = [t[1] for t in tests if t[1] is not None and t[1] > 0]
            gdict['avg_sigma_debond'] = float(np.mean(debonds)) if debonds else None
            gdict['avg_sigma_ultimate'] = float(np.mean(ultimates)) if ultimates else None
        result.append(gdict)

    conn.close()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return output_path


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    import sys

    init_db()
    print(f"Database: {DB_PATH}")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python data_manager.py seed       - 从JSON初始化数据库")
        print("  python data_manager.py template   - 导出CSV导入模板")
        print("  python data_manager.py import CSV - 导入CSV实验数据")
        print("  python data_manager.py stats      - 查看统计")
        print("  python data_manager.py export     - 导出完整JSON")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'seed':
        seed_from_json()
    elif cmd == 'template':
        path = export_csv_template()
        print(f"Template saved to: {path}")
    elif cmd == 'import':
        if len(sys.argv) < 3:
            print("Usage: python data_manager.py import <path.csv>")
            sys.exit(1)
        import_csv(sys.argv[2])
    elif cmd == 'stats':
        stats = get_statistics()
        for k, v in stats.items():
            print(f"  {k}: {v}")
    elif cmd == 'export':
        path = export_full_json()
        print(f"Exported to: {path}")
    else:
        print(f"Unknown command: {cmd}")