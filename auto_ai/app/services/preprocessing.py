import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class PreprocessingPipeline:
    def __init__(self, strategy: Dict[str, Any]):
        self.strategy = strategy
        self.scaler_type = strategy.get("scaler", "StandardScaler")
        self.encoder_type = strategy.get("categorical_encoder", "OneHotEncoder")
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.log_transformed_cols = []
        
    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        
        features = [c for c in df.columns if c != target_col]
        self.numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_cols = [c for c in features if c not in self.numeric_cols]
        
        # 1. Imputation
        for col in self.numeric_cols:
            median = df[col].median()
            self.imputers[col] = median
            df[col] = df[col].fillna(median)
            
        for col in self.categorical_cols:
            mode = df[col].mode()
            mode_val = mode.iloc[0] if not mode.empty else "Missing"
            self.imputers[col] = mode_val
            df[col] = df[col].fillna(mode_val)
            
        # 2. Skewness Correction
        for col in self.numeric_cols:
            skew = df[col].skew()
            if abs(skew) > 1.0:
                min_val = df[col].min()
                if min_val >= 0:
                    df[col] = np.log1p(df[col])
                    self.log_transformed_cols.append(col)
                    
        # 3. Scaling
        if self.scaler_type == "RobustScaler":
            scaler_inst = RobustScaler()
        elif self.scaler_type == "MinMaxScaler":
            scaler_inst = MinMaxScaler()
        else:
            scaler_inst = StandardScaler()
            
        if self.numeric_cols:
            df[self.numeric_cols] = scaler_inst.fit_transform(df[self.numeric_cols])
            self.scalers["numeric"] = scaler_inst
            
        # 4. Encoding
        if self.encoder_type == "TargetEncoder" and self.categorical_cols:
            y = df[target_col]
            for col in self.categorical_cols:
                means = y.groupby(df[col]).mean()
                self.encoders[col] = means.to_dict()
                df[col] = df[col].map(self.encoders[col]).fillna(y.mean())
        elif self.categorical_cols:
            for col in self.categorical_cols:
                cats = df[col].unique()
                mapping = {cat: idx for idx, cat in enumerate(cats)}
                self.encoders[col] = mapping
                df[col] = df[col].map(mapping).fillna(-1)
                
        return df

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Imputation
        for col in self.numeric_cols:
            df[col] = df[col].fillna(self.imputers.get(col, 0.0))
            
        for col in self.categorical_cols:
            df[col] = df[col].fillna(self.imputers.get(col, "Missing"))
            
        # 2. Skewness Correction
        for col in self.log_transformed_cols:
            df[col] = np.log1p(df[col].clip(lower=0))
                    
        # 3. Scaling
        if "numeric" in self.scalers and self.numeric_cols:
            df[self.numeric_cols] = self.scalers["numeric"].transform(df[self.numeric_cols])
            
        # 4. Encoding
        if self.categorical_cols:
            for col in self.categorical_cols:
                mapping = self.encoders.get(col, {})
                if isinstance(mapping, dict):
                    # Use map, fill unknown categories with default 0
                    df[col] = df[col].map(mapping).fillna(0.0)
                    
        return df
