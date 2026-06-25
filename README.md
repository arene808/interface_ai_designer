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
├── ai_model.py                    # AI核心模型（前向预测+逆向设计）
├── data_manager.py                # SQLite数据管理（CSV导入/导出/查询）
├── fourier_interface_analysis.py  # 界面曲线傅里叶变换统一表示
├── app.py                         # Flask Web API服务器
├── interface_fourier_features.json # 12组界面傅里叶特征数据
├── import_template.csv            # 实验数据CSV导入模板
├── requirements.txt
├── templates/
│   └── index.html                 # Web可视化UI
└── static/                        # 静态资源
```

## 数据导入流程

1. 做拉伸试验得到 σ_debond
2. 按 `import_template.csv` 格式整理数据
3. Web UI上传CSV → 自动入库
4. 点击"用真实数据重新训练"
5. 拖动滑块 → AI输出最优曲线

## 界面类型

| 类型 | 强度范围 | 特点 |
|------|---------|------|
| 平面 Flat | 4.5~5.8 MPa | 纯粘附，零互锁 |
| 正弦 Sinusoidal | 5.5~12.0 MPa | 连续曲率，无尖角 |
| 锯齿 Sawtooth | 6.5~13.5 MPa | 尖角互锁 |
| 燕尾 Dovetail | 8.0~16.0 MPa | 机械锁扣，最强 |

## 技术栈

Python · Flask · scikit-learn · NumPy · SQLite · Canvas API