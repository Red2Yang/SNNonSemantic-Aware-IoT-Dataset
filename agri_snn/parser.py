import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch


class AgriDatasetParser:
    NUMERIC_FEATURES = ['N', 'P', 'K', 'Moisture', 'pH',
                        'Temperature', 'Humidity']
    TIMESTAMP_COL = 'UAV_Timestamp'

    ACTION_CLASSES = ['Apply Fertilizer', 'Apply Pesticide',
                      'Irrigate', 'Monitor']

    def __init__(self, xlsx_path: str, sheet_name: str = 'Sheet1'):
        self.xlsx_path = xlsx_path
        self.sheet_name = sheet_name
        self.df = None

        self.scaler = StandardScaler() 
        self.label_encoders = {}
        self.action_class_weights = None

        self.ms_cols = []
        self.rgb_cols = []
        self.input_dims = {}

    def load(self) -> pd.DataFrame:
        self.df = pd.read_excel(self.xlsx_path, sheet_name=self.sheet_name)
        print(f"[数据加载] {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
        return self.df

    def parse_timestamps(self):
        self.df[self.TIMESTAMP_COL] = pd.to_datetime(
            self.df[self.TIMESTAMP_COL], format='%m/%d/%Y %H:%M'
        )
        self.df = self.df.sort_values(
            ['Zone_ID', self.TIMESTAMP_COL]
        ).reset_index(drop=True)
        print(f"[时间戳] {self.df[self.TIMESTAMP_COL].min()} → "
              f"{self.df[self.TIMESTAMP_COL].max()}")

    def fix_semantic_tags(self):
        def clean(tag):
            tags = set(t.strip() for t in str(tag).split(',') if t.strip())
            return ', '.join(sorted(tags))
        self.df['Semantic_Tag'] = self.df['Semantic_Tag'].apply(clean)

    def fix_nutrient_logic(self):
        def revise(row):
            tags = set()
            existing = set(row['Semantic_Tag'].split(', '))
            
            tags = existing - {'N-deficiency', 'P-deficiency',
                               'K-deficiency', 'Healthy'}
            if row['N'] < 40:
                tags.add('N-deficiency')
            if row['P'] < 30:
                tags.add('P-deficiency')
            if row['K'] < 50:
                tags.add('K-deficiency')
            if not tags.intersection({'N-deficiency', 'P-deficiency',
                                      'K-deficiency', 'Pest-risk'}):
                tags.add('Healthy')
            else:
                tags.discard('Healthy')
            return ', '.join(sorted(tags))
        self.df['Semantic_Tag'] = self.df.apply(revise, axis=1)

    def fix_action_logic(self):
        def rule(row):
            if row['Temperature'] > 30 and row['Humidity'] > 70:
                return 'Apply Pesticide'
            if row['N'] < 40 or row['P'] < 30 or row['K'] < 50:
                return 'Apply Fertilizer'
            if row['Moisture'] < 15:
                return 'Irrigate'
            return 'Monitor'

        self.df['Action_Suggested'] = self.df.apply(rule, axis=1)
        dist = self.df['Action_Suggested'].value_counts()
        print("[动作标签] 分布：")
        for act, cnt in dist.items():
            print(f"  {act}: {cnt} ({cnt / len(self.df) * 100:.1f}%)")

    def build_features(self):
        ms_mask = self.df['Image_Type'] == 'Multispectral'
        rgb_mask = self.df['Image_Type'] == 'RGB'

        self.df['NDVI_clean'] = self.df['NDVI'].where(ms_mask, np.nan)
        self.df['NDRE_clean'] = self.df['NDRE'].where(ms_mask, np.nan)
        self.df['RGB_Damage_clean'] = self.df['RGB_Damage_Score'].where(
            rgb_mask, np.nan
        )

        common_cols = list(self.NUMERIC_FEATURES)
        self.df[common_cols] = self.scaler.fit_transform(
            self.df[common_cols]
        )

        le = LabelEncoder()
        self.df['action_label'] = le.fit_transform(
            self.df['Action_Suggested']
        )
        self.label_encoders['Action_Suggested'] = le

        classes = np.unique(self.df['action_label'])
        weights = compute_class_weight('balanced',
                                       classes=classes,
                                       y=self.df['action_label'])
        self.action_class_weights = torch.FloatTensor(weights)
        print(f"[类别权重] {dict(zip(le.classes_, np.round(weights, 3)))}")

        self.ms_cols = ['NDVI_clean', 'NDRE_clean'] + common_cols
        self.rgb_cols = ['RGB_Damage_clean'] + common_cols

        X_ms = self.df[self.ms_cols].values.astype(np.float32)
        X_rgb = self.df[self.rgb_cols].values.astype(np.float32)
        y_action = self.df['action_label'].values

        self.input_dims = {
            'ms': X_ms.shape[1],
            'rgb': X_rgb.shape[1],
        }
        print(f"[特征矩阵] MS: {X_ms.shape}, RGB: {X_rgb.shape}")
        return X_ms, X_rgb, y_action

    def run_pipeline(self):
        print("=" * 60)
        print("  无人机决策")
        print("=" * 60)
        self.load()
        self.parse_timestamps()
        self.fix_semantic_tags()
        self.fix_nutrient_logic()
        self.fix_action_logic()
        X_ms, X_rgb, y_action = self.build_features()
        print("=" * 60)
        print("  处理完成")
        print("=" * 60)
        return X_ms, X_rgb, y_action