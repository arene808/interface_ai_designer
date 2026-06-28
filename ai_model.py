# -*- coding: utf-8 -*-
"""
精益求径 - AI逆设计模型
========================
输入: 需求强度(目标界面脱粘强度 sigma_debond, MPa)
输出: 最优傅里叶系数 -> 界面曲线路径 (x, y)

架构:
  1. 物理信息合成数据生成器(在无实验数据时训练模型)
  2. 前向模型: 傅里叶特征 -> sigma_debond(MLP回归器)
  3. 逆向模型: sigma_debond -> 傅里叶特征 -> 曲线重建
  4. 可制造性约束过滤器
"""

import numpy as np
import json
import os
import sys

SPECIMEN_WIDTH = 13.0
PERIOD = 4.8
N_HARMONICS = 5
N_SAMPLE_POINTS = 200
FLAT_END_LENGTH = 1.7
MIN_FEATURE_SIZE = 1.2
FEATURE_DIM = 8


class SyntheticDataGenerator:
    """基于物理原理生成合成训练数据"""
    
    BASE_ADHESION = 5.0
    INTERLOCK_COEFF = 3.2
    TORTUOSITY_COEFF = 4.5
    CURVATURE_PENALTY = 0.08
    TYPE_BONUS = {
        'flat': 0.0,
        'sinusoidal': 1.2,
        'sawtooth': 2.0,
        'dovetail': 3.5,
    }
    NOISE_LEVEL = 0.3
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
    
    def compute_strength(self, features, interface_type):
        A1, A2, A3, A4, A5, rms, curv_max, tort = features
        sigma = self.BASE_ADHESION
        asymmetry = (abs(A2) + abs(A4)) * 0.8
        interlock = self.INTERLOCK_COEFF * (A1 + asymmetry)
        sigma += interlock
        sigma += self.TORTUOSITY_COEFF * (tort - 1.0)
        sigma += self.TYPE_BONUS.get(interface_type, 0)
        penalty = self.CURVATURE_PENALTY * curv_max * (1.0 / (1.0 + np.exp(-(curv_max - 30) / 15)))
        sigma -= penalty
        if A1 > 2.5:
            sigma -= 0.8 * (A1 - 2.5) ** 1.5
        sigma += self.rng.randn() * self.NOISE_LEVEL
        return max(sigma, 0.5)
    
    def generate_dataset(self, n_samples_per_type=200):
        data = []
        type_ranges = {
            'flat': {
                'A1': (0, 0.05), 'A2': (0, 0.02), 'A3': (0, 0.02),
                'A4': (0, 0.01), 'A5': (0, 0.01),
                'rms': (0, 0.02), 'curv_max': (0, 1.0), 'tort': (1.0, 1.005),
            },
            'sinusoidal': {
                'A1': (0.2, 2.5), 'A2': (0, 0.15), 'A3': (0.01, 0.12),
                'A4': (0, 0.06), 'A5': (0, 0.05),
                'rms': (0.1, 1.6), 'curv_max': (10, 60), 'tort': (1.05, 2.1),
            },
            'sawtooth': {
                'A1': (0.15, 2.5), 'A2': (0, 0.12), 'A3': (0.01, 0.15),
                'A4': (0, 0.09), 'A5': (0.01, 0.08),
                'rms': (0.1, 1.3), 'curv_max': (20, 140), 'tort': (1.05, 2.1),
            },
            'dovetail': {
                'A1': (0.6, 2.2), 'A2': (0.1, 1.2), 'A3': (0.05, 0.3),
                'A4': (0.02, 0.3), 'A5': (0.01, 0.25),
                'rms': (0.5, 1.9), 'curv_max': (30, 85), 'tort': (1.15, 1.8),
            },
        }
        for itype, ranges in type_ranges.items():
            for _ in range(n_samples_per_type):
                features = np.array([
                    self.rng.uniform(*ranges['A1']),
                    self.rng.uniform(*ranges['A2']),
                    self.rng.uniform(*ranges['A3']),
                    self.rng.uniform(*ranges['A4']),
                    self.rng.uniform(*ranges['A5']),
                    self.rng.uniform(*ranges['rms']),
                    self.rng.uniform(*ranges['curv_max']),
                    self.rng.uniform(*ranges['tort']),
                ])
                strength = self.compute_strength(features, itype)
                data.append({
                    'features': features,
                    'strength': strength,
                    'type': itype,
                })
        return data


class InterfaceAIModel:
    """界面AI代理模型: 前向预测 + 逆向设计"""
    
    def __init__(self):
        self.forward_model = None
        self.inverse_models = {}
        self.scaler_X = None
        self.scaler_y = None
        self.is_trained = False
        self.training_data = None
        self.inverse_data = {}
        self.reference_groups = self._load_reference_groups()
    
    def _load_reference_groups(self):
        return {
            'F0': {'features': [0, 0, 0, 0, 0, 0, 0, 1.001], 'type': 'flat', 'amp': 0},
            'N1': {'features': [0.239, 0.616, 0.779, 0.037, 0.097, 0.729, 36.57, 1.343], 'type': 'sinusoidal', 'amp': 1.2},
            'N2': {'features': [0.359, 0.925, 1.169, 0.055, 0.146, 1.093, 47.99, 1.640], 'type': 'sinusoidal', 'amp': 1.8},
            'N3': {'features': [0.479, 1.233, 1.558, 0.074, 0.195, 1.458, 54.03, 1.964], 'type': 'sinusoidal', 'amp': 2.4},
            'S1': {'features': [0.189, 0.492, 0.637, 0.031, 0.089, 0.595, 69.01, 1.304], 'type': 'sawtooth', 'amp': 1.2},
            'S2': {'features': [0.283, 0.738, 0.955, 0.047, 0.133, 0.893, 99.58, 1.589], 'type': 'sawtooth', 'amp': 1.8},
            'S3': {'features': [0.377, 0.984, 1.273, 0.063, 0.178, 1.190, 125.98, 1.907], 'type': 'sawtooth', 'amp': 2.4},
            'D1': {'features': [0.822, 0.532, 0.159, 0.224, 0.000, 0.751, 53.43, 1.316], 'type': 'dovetail', 'amp': 1.2},
            'D2': {'features': [1.429, 0.838, 0.196, 0.084, 0.121, 1.232, 64.91, 1.497], 'type': 'dovetail', 'amp': 1.8},
            'D3': {'features': [2.048, 1.117, 0.210, 0.107, 0.234, 1.722, 75.94, 1.680], 'type': 'dovetail', 'amp': 2.4},
        }
    
    def train(self, n_synthetic=300):
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_dataset(n_samples_per_type=n_synthetic)
        
        X = np.array([d['features'] for d in data])
        y = np.array([d['strength'] for d in data])
        types = [d['type'] for d in data]
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        self.forward_model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu', solver='adam', alpha=0.001,
            batch_size=32, learning_rate='adaptive',
            max_iter=2000, early_stopping=False,
            random_state=42,
        )
        self.forward_model.fit(X_scaled, y_scaled)
        
        from sklearn.linear_model import Ridge
        for itype in ['flat', 'sinusoidal', 'sawtooth', 'dovetail']:
            mask = [t == itype for t in types]
            if sum(mask) < 10:
                continue
            X_type = X[mask]
            y_type = y[mask]
            models = []
            for j in range(FEATURE_DIM):
                y_j = X_type[:, j]
                model_j = Ridge(alpha=0.1)
                s = y_type.reshape(-1, 1)
                s_poly = np.hstack([s, s**2, s**3])
                model_j.fit(s_poly, y_j)
                models.append(model_j)
            self.inverse_models[itype] = models
        
        self.is_trained = True
        self.training_data = data
        self.strength_min = float(np.min(y))
        self.strength_max = float(np.max(y))
        
        return {
            'status': 'trained',
            'n_samples': len(data),
            'strength_range': [self.strength_min, self.strength_max],
            'forward_score': self.forward_model.score(X_scaled, y_scaled),
        }
    
    def predict_strength(self, features):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.forward_model.predict(X_scaled)
        return float(self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0, 0])
    
    def inverse_design(self, target_strength, preferred_type=None):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        target = np.clip(target_strength, self.strength_min, self.strength_max)
        
        if preferred_type and preferred_type in self.inverse_models:
            best_type = preferred_type
        else:
            best_type = self._select_best_type(target)
        
        models = self.inverse_models.get(best_type)
        if models is None:
            best_type = 'sinusoidal'
            models = self.inverse_models[best_type]
        s = np.array([[target, target**2, target**3]])
        features = np.zeros(FEATURE_DIM)

        # Type-specific interpolation for small datasets
        if (hasattr(self, 'inverse_data') and best_type in self.inverse_data and len(self.inverse_data[best_type]['strengths']) >= 1):
            idata = self.inverse_data[best_type]
            strengths = idata['strengths']
            feats = idata['features']
            if len(strengths) == 1:
                features = feats[0].copy()
            else:
                idx = np.argsort(strengths)
                s_sorted = strengths[idx]
                f_sorted = feats[idx]
                if target <= s_sorted[0]:
                    features = f_sorted[0].copy()
                elif target >= s_sorted[-1]:
                    features = f_sorted[-1].copy()
                else:
                    for k in range(len(s_sorted) - 1):
                        if s_sorted[k] <= target <= s_sorted[k+1]:
                            t = (target - s_sorted[k]) / (s_sorted[k+1] - s_sorted[k])
                            features = (1 - t) * f_sorted[k] + t * f_sorted[k+1]
                            break
        else:
            for j, model in enumerate(models):
                features[j] = max(0, model.predict(s)[0])

        if best_type == 'flat':
            features = np.zeros(FEATURE_DIM)
            features[7] = 1.001

        predicted_strength = self.predict_strength(features)
        curve_params = self._features_to_curve_params(features, best_type)
        curve_x, curve_y = self._generate_curve_from_params(curve_params)

        return {
            'target_strength': float(target_strength),
            'predicted_strength': float(predicted_strength),
            'curve_type': best_type,
            'curve_type_cn': self._type_name_cn(best_type),
            'features': features.tolist(),
            'feature_labels': ['A1(mm)', 'A2(mm)', 'A3(mm)', 'A4(mm)', 'A5(mm)',
                               'RMS(mm)', 'kmax(1/mm)', 'tau'],
            'curve_params': curve_params,
            'curve_x': curve_x.tolist(),
            'curve_y': curve_y.tolist(),
            'strength_range': [self.strength_min, self.strength_max],
        }
    

    
    def _select_best_type(self, target):
        ranges = {
            'flat':       (1.0, 1.6),
            'sinusoidal': (1.4, 2.6),
            'sawtooth':   (1.6, 2.5),
            'dovetail':   (2.0, 3.5),
        }
        for itype, (lo, hi) in ranges.items():
            if lo <= target <= hi:
                return itype
        best = 'sinusoidal'
        best_dist = float('inf')
        for itype, (lo, hi) in ranges.items():
            dist = min(abs(target - lo), abs(target - hi))
            if dist < best_dist:
                best_dist = dist
                best = itype
        return best
    
    def _features_to_curve_params(self, features, curve_type):
        A1 = features[0]
        if curve_type == 'flat':
            return {'interface_type': 'flat', 'amplitude': 0.0, 'period': PERIOD}
        elif curve_type == 'sinusoidal':
            amp = np.clip(A1 * 1.05, 0.3, 2.5)
            return {'interface_type': 'sinusoidal', 'amplitude': float(amp), 'period': PERIOD}
        elif curve_type == 'sawtooth':
            amp = np.clip(A1 * 1.3, 0.3, 2.5)
            return {'interface_type': 'sawtooth', 'amplitude': float(amp), 'period': PERIOD, 'sawtooth_asymmetry': 0.5}
        elif curve_type == 'dovetail':
            depth = np.clip(A1 * 0.85, 0.5, 2.5)
            if depth < 1.2:
                nw, hw = 2.0, 2.4
            elif depth < 1.8:
                nw, hw = 1.5, 2.9
            else:
                nw, hw = 1.2, 3.2
            return {'interface_type': 'dovetail', 'amplitude': float(depth), 'period': PERIOD,
                    'neck_width': float(nw), 'head_width': float(hw), 'dovetail_depth': float(depth)}
        return {'interface_type': 'sinusoidal', 'amplitude': 1.5, 'period': PERIOD}
    
    def _generate_curve_from_params(self, params):
        itype = params['interface_type']
        amplitude = params.get('amplitude', 0)
        period = params.get('period', PERIOD)
        x = np.linspace(0, SPECIMEN_WIDTH, N_SAMPLE_POINTS)
        omega = 2 * np.pi / period
        
        if itype == 'flat':
            y = np.zeros_like(x)
        elif itype == 'sinusoidal':
            phi = np.pi - omega * FLAT_END_LENGTH
            y = amplitude * np.sin(omega * x + phi)
        elif itype == 'sawtooth':
            asym = params.get('sawtooth_asymmetry', 0.5)
            phi = np.pi * asym - omega * FLAT_END_LENGTH
            y = np.zeros_like(x)
            for i in range(len(x)):
                phase = (omega * x[i] + phi) % (2 * np.pi)
                np_val = phase / (2 * np.pi)
                if np_val < asym:
                    y[i] = amplitude * (2 * np_val / asym - 1)
                else:
                    y[i] = amplitude * (1 - 2 * (np_val - asym) / (1 - asym))
        elif itype == 'dovetail':
            p = period
            nw = params.get('neck_width', 1.5)
            hw = params.get('head_width', 2.9)
            d = params.get('dovetail_depth', 1.8)
            sw = max((p - nw - hw) / 2.0, 0.1)
            zero_pos = nw / 2.0
            zero_norm = zero_pos / p
            phi = 2.0 * np.pi * zero_norm - omega * FLAT_END_LENGTH
            seg1 = (nw / 2.0) / p
            seg2 = (nw / 2.0 + sw) / p
            seg3 = (nw / 2.0 + sw + hw) / p
            seg4 = (nw / 2.0 + sw + hw + sw) / p
            y = np.zeros_like(x)
            for i in range(len(x)):
                phase = (omega * x[i] + phi) % (2.0 * np.pi)
                norm = phase / (2.0 * np.pi)
                unit_index = int((omega * x[i] + phi) // (2.0 * np.pi))
                sign = 1.0 if (unit_index % 2 == 0) else -1.0
                if norm < seg1:
                    y[i] = 0.0
                elif norm < seg2:
                    frac = (norm - seg1) / (seg2 - seg1)
                    y[i] = sign * d * (1.0 - np.cos(np.pi * frac)) / 2.0
                elif norm < seg3:
                    y[i] = sign * d
                elif norm < seg4:
                    frac = (norm - seg3) / (seg4 - seg3)
                    y[i] = sign * d * (1.0 + np.cos(np.pi * frac)) / 2.0
                else:
                    y[i] = 0.0
        else:
            y = np.zeros_like(x)
        
        mask_left = x < FLAT_END_LENGTH
        mask_right = x > (SPECIMEN_WIDTH - FLAT_END_LENGTH)
        y = np.where(mask_left | mask_right, 0, y)
        return x, y
    
    def _type_name_cn(self, itype):
        names = {
            'flat': '平面界面 (Flat)',
            'sinusoidal': '正弦界面 (Sinusoidal)',
            'sawtooth': '锯齿界面 (Sawtooth)',
            'dovetail': '燕尾界面 (Dovetail)',
        }
        return names.get(itype, itype)
    
    def get_strength_range(self):
        if self.is_trained:
            return [self.strength_min, self.strength_max]
        return [3.0, 18.0]
    
    def get_reference_curves(self):
        curves = []
        for name, info in self.reference_groups.items():
            params = self._features_to_curve_params(
                np.array(info['features']), info['type'])
            x, y = self._generate_curve_from_params(params)
            curves.append({
                'name': name,
                'type': info['type'],
                'type_cn': self._type_name_cn(info['type']),
                'amplitude': info['amp'],
                'x': x.tolist(),
                'y': y.tolist(),
            })
        return curves



    def train_from_real_data(self, X_real, y_real):
        """使用真实实验数据训练（替代或补充合成数据）
        
        参数:
          X_real: np.ndarray, shape (n_samples, 8), 傅里叶特征矩阵
          y_real: np.ndarray, shape (n_samples,),  脱粘强度 (MPa)
        
        返回:
          dict with training metrics
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import cross_val_score
        
        GROUP_TYPE_MAP = {
            'F0': 'flat', 'F1': 'flat',
            'N1': 'sinusoidal', 'N2': 'sinusoidal', 'N3': 'sinusoidal',
            'S1': 'sawtooth', 'S2': 'sawtooth', 'S3': 'sawtooth',
            'D1': 'dovetail', 'D2': 'dovetail', 'D3': 'dovetail',
        # GROUP_TYPE_MAP already defined above
        
        
        
        
        
        }
        if len(X_real) < 3:
            return {
                'status': 'error',
                'message': f'需要至少5组真实数据，当前只有{len(X_real)}组。建议先用合成数据预训练。',
            }
        
        # 归一化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X_real)
        y_scaled = self.scaler_y.fit_transform(y_real.reshape(-1, 1)).ravel()
        
        # 训练前向模型（真实数据量小时用更强正则化）
        alpha = 0.01 if len(X_real) < 20 else 0.001
        self.forward_model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu', solver='adam', alpha=alpha,
            batch_size=min(16, len(X_real)),
            learning_rate='adaptive',
            max_iter=3000, early_stopping=False,
            random_state=42,
        )
        self.forward_model.fit(X_scaled, y_scaled)
        
        # 交叉验证评估
        try:
            cv_scores = cross_val_score(
                self.forward_model, X_scaled, y_scaled,
                cv=min(5, len(X_real)), scoring='r2'
            )
            cv_r2 = float(np.mean(cv_scores))
        except Exception:
            cv_r2 = None
        
        # Store type-specific data for interpolation-based inverse design
        # Derive type from feature pattern: flat(A1=0), sinusoidal(A2 large), sawtooth(curv high), dovetail(A2 large+A4 large)
        self.inverse_data = {}
        for i in range(len(y_real)):
            feat = X_real[i]
            A1, A2, A4, curv = feat[0], feat[1], feat[3], feat[6]
            if A1 < 0.01:
                itype = 'flat'
            elif A4 > A2 * 0.3:
                itype = 'dovetail'
            elif curv > 60:
                itype = 'sawtooth'
            else:
                itype = 'sinusoidal'
            if itype not in self.inverse_data:
                self.inverse_data[itype] = {'strengths': [], 'features': []}
            self.inverse_data[itype]['strengths'].append(y_real[i])
            self.inverse_data[itype]['features'].append(X_real[i])
        for itype in self.inverse_data:
            self.inverse_data[itype]['strengths'] = np.array(self.inverse_data[itype]['strengths'])
            self.inverse_data[itype]['features'] = np.array(self.inverse_data[itype]['features'])
        
        # 训练逆模型
        from sklearn.linear_model import Ridge
        self.inverse_models = {}
        valid_types = ['flat', 'sinusoidal', 'sawtooth', 'dovetail']
        
        for itype in valid_types:
            models = []
            for j in range(FEATURE_DIM):
                model_j = Ridge(alpha=0.5)  # 强正则化
                s = y_real.reshape(-1, 1)
                s_poly = np.hstack([s, s**2, s**3])
                model_j.fit(s_poly, X_real[:, j])
                models.append(model_j)
            self.inverse_models[itype] = models
        
        self.is_trained = True
        self.strength_min = float(np.min(y_real))
        self.strength_max = float(np.max(y_real))
        
        train_score = self.forward_model.score(X_scaled, y_scaled)
        
        return {
            'status': 'trained_on_real_data',
            'n_samples': len(X_real),
            'strength_range': [self.strength_min, self.strength_max],
            'train_r2': float(train_score),
            'cv_r2': cv_r2,
            'warning': '数据量较少，预测外推能力有限' if len(X_real) < 15 else None,
        }

_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = InterfaceAIModel()
        _model_instance.train(n_synthetic=300)
    return _model_instance


if __name__ == '__main__':
    print("=" * 60)
    print("精益求径 - AI逆设计模型 测试")
    print("=" * 60)
    model = get_model()
    print(f"\n模型状态: trained={model.is_trained}")
    print(f"强度范围: {model.strength_min:.1f} ~ {model.strength_max:.1f} MPa")
    for s in [5.0, 8.0, 11.0, 14.0]:
        print(f"\n{'─'*40}")
        print(f"需求强度: {s} MPa")
        result = model.inverse_design(s)
        print(f"  推荐类型: {result['curve_type_cn']}")
        print(f"  预测强度: {result['predicted_strength']:.2f} MPa")
        print(f"  特征向量: {[f'{v:.3f}' for v in result['features']]}")
    print("\n" + "=" * 60)
    print("测试完成!")
