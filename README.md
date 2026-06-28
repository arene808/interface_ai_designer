# interface_ai_designer

**AI+AM** — 多材料3D打印界面AI逆设计引擎

输入需求强度（σ_debond, MPa），AI模型输出最优界面曲线路径，用于TPU-PLA异质材料3D打印互锁界面设计。

## 功能

- **AI逆设计**：目标强度 → 傅里叶特征向量 → 界面曲线几何
- **多类型支持**：平面/正弦/锯齿/燕尾四种界面
- **数据管理**：SQLite数据库，支持CSV批量导入实验数据
- **真实数据训练**：用拉伸试验数据重训练模型
- **可视化UI**：Web界面，实时滑块调节 + Canvas曲线预览

## 快速开始

```bash
pip install -r requirements.txt
python app.py
```

打开 http://127.0.0.1:5000

## 文件结构

```
interface_ai_designer/
├── ai_model.py                     # AI核心模型（前向预测+逆向设计）
├── data_manager.py                 # SQLite数据管理（CSV导入/导出/查询）
├── fourier_interface_analysis.py   # 界面曲线傅里叶变换统一表示
├── analyze_test_data.py            # 真实拉伸数据一键分析脚本
├── app.py                          # Flask Web API服务器
├── interface_fourier_features.json # 12组界面傅里叶特征数据（含实验值）
├── import_template.csv             # 实验数据CSV导入模板
├── model_config.json               # 模型训练配置
├── requirements.txt
├── analysis_output/                # 分析输出（图表+数据）
│   ├── chart_strength_comparison.png
│   ├── chart_stress_strain_curves.png
│   ├── chart_feature_vs_strength.png
│   ├── chart_amplitude_effect.png
│   ├── training_database.json
│   └── training_data.csv
├── templates/
│   └── index.html                  # Web可视化UI
└── static/                         # 静态资源
```

## 数据导入流程

1. 做拉伸试验得到 σ_debond
2. 按 `import_template.csv` 格式整理数据
3. Web UI上传CSV → 自动入库
4. 点击"用真实数据重新训练"
5. 拖动滑块 → AI输出最优曲线

## 界面类型与实测数据

### 真实测试数据 (2026-06-27, GB/T 528-2009, 50mm/min)

| 类型 | 组别 | 振幅 | 实测强度 (MPa) | CV | 断裂模式 |
|------|------|------|---------------|-----|---------|
| 平面 Flat | F0 | 0 | **1.36 ± 0.22** | 16.0% | AF |
| 平面 Flat | F1 | 0 | 1.16 ± 0.22 | 19.4% | AF |
| 正弦 Sin | N1 | 1.2mm | 1.92 ± 0.24 | 12.3% | CF-T |
| 正弦 Sin | N2 | 1.8mm | 1.51 ± 0.11 ⚠️ | 7.5% | AF |
| 正弦 Sin | N3 | 2.4mm | **2.38 ± 0.21** 🏆 | 8.8% | CF-T |
| 锯齿 Saw | S1 | 1.2mm | 1.90 ± 0.22 | 11.5% | CF-T |
| 锯齿 Saw | S2 | 1.8mm | 1.99 ± 0.16 | 7.9% | CF-T |
| 锯齿 Saw | S3 | 2.4mm | 2.23 ± 0.20 | 9.0% | CF-T |

> ⚠️ N2 (正弦A=1.8mm) 强度异常低于N1，待排查。燕尾(D)系列待测试。

### AI模型推荐范围

| 类型 | 推荐强度范围 | 特点 |
|------|-------------|------|
| 平面 Flat | 1.0~1.6 MPa | 纯粘附，零互锁 |
| 正弦 Sinusoidal | 1.4~2.6 MPa | 连续曲率，无尖角 |
| 锯齿 Sawtooth | 1.6~2.5 MPa | 尖角互锁 |
| 燕尾 Dovetail | 2.0~3.5 MPa | 机械锁扣，最强（待实测） |

## 技术栈

Python · Flask · scikit-learn · NumPy · SQLite · Canvas API

---

## 版本更新

### v2.0 — 真实数据驱动 (2026-06-28)

- ✅ **导入35条真实拉伸测试数据** (F0/F1/N1/N2/N3/S1/S2/S3 共8组)
- ✅ **AI模型重训练**：前向预测 R² = 0.994，强度范围 1.0~3.5 MPa
- ✅ **逆设计升级**：从 Ridge 回归改为**类型特化插值**，解决小样本逆推问题
- ✅ **4张分析图表**：强度对比柱状图、应力应变曲线、特征-强度散点图、振幅效应图
- ✅ **UI适配真实数据**：滑块范围调整为 0.8~3.5 MPa
- ✅ **SQLite数据库**：35条记录，支持CSV导入/手动录入/JSON导出
- ⚠️ **已知限制**：燕尾(D)系列尚无实测数据，使用外推估计；N2数据异常待排查

### v1.0 — 合成数据原型 (2026-06-25)

- 基于物理信息合成数据训练 (σ = 5~17 MPa)
- MLP前向 + Ridge逆模型架构
- 4种界面类型支持（平面/正弦/锯齿/燕尾）
- Flask Web可视化UI
- 傅里叶统一表示 (8D特征向量)