"""
Author: Jet Deng
Date: 2024-02-21 10:08:15
LastEditTime: 2024-03-01 16:35:32
Description: ML工具类
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Union
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

def rolling_pca(
    factor_df: pd.DataFrame, window: int, n_components: int
) -> pd.DataFrame:
    """对所有品种的因子进行滚动窗口PCA分解
    Args:
        factor_df (pd.DataFrame): columns=[factor_names], index=datetime
        window (int):
        n_components (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    factor_df_list = []
    idx = []
    for i in range(factor_df.shape[0] - window + 1):
        window_df = factor_df.iloc[i : i + window, :].ffill().fillna(0)
        pca = PCA(n_components=n_components)
        output = pca.fit_transform(window_df)

        factor_df_list.append(output[-1, :])
        idx.append(window_df.index[-1])
    res = pd.DataFrame(
        factor_df_list, index=idx, columns=[f"pca_{i}" for i in range(n_components)]
    )
    res.index = idx
    return res


class MLTrainFit:
    """滚动训练模型"""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        model: object | str,
        train_start: str | datetime,
        train_end: str | datetime,
        predict_window: str,
        train_method: str = 'none',
        **kwargs,
    ):
        self.X = X  # two-level index with datetime and product, obtained from PrepareXY
        self.y = y  # two-level index with datetime and product, obtained from PrepareXY
        self.model = model
        self.train_method = train_method
        self.train_start = pd.to_datetime(train_start) if isinstance(train_start, str) else train_start
        self.train_end = pd.to_datetime(train_end) if isinstance(train_end, str) else train_end
        self.predict_window = predict_window
        
        if self.predict_window.endswith("d"):
            self.test_start = pd.to_datetime(kwargs.get("test_start", train_start + timedelta(days=int(predict_window[:-1]))))
        elif self.predict_window.endswith("m"):
            self.test_start = pd.to_datetime(kwargs.get("test_start", train_start + timedelta(minutes=int(predict_window[:-1]))))
        else:
            raise ValueError("Predict Window Not Supported!")
        
        
        
        self.retrain_freq = kwargs.get(
            "freq", "6M"
        )  # Rolling/Expanding Window
        

        self.model_params = kwargs.get("model_params", {})

    def _model(self):
        """Initialize model"""
        if isinstance(self.model, str):
            model = eval(self.model)(**self.model_params)
        else:
            model = self.model
        return model
    
    def fit_predict(self):
        if self.train_method == 'none':
            return self._fit_predict()
        
        elif self.train_method == "rolling":
            return self._rolling_fit_predict()
        
        elif self.train_method == "expanding":
            return self._expanding_fit_predict()
        
        else:
            raise ValueError("Train Method Not Supported!")

    def _split_train_test(self, X: pd.DataFrame, y: pd.DataFrame, train_start: str, train_end: str, test_start: str, test_end: str):
        """Split train and test data based on the start point and end point"""
        X, y = X.align(y, axis=0, join="inner")
        
        # convert the multi-index to columns
        X = X.reset_index()
        y = y.reset_index()

        # split the data
        if "date" in X.columns:
            self.time_col = "date"
            train_X = X[(X["date"] >= train_start) & (X["date"] < train_end)]
            train_y = y[(y["date"] >= train_start) & (y["date"] < train_end)]
            test_X = X[(X["date"] >= test_start) & (X["date"] < test_end)]
            test_y = y[(y["date"] >= test_start) & (y["date"] < test_end)]
        elif "datetime" in X.columns:
            self.time_col = "datetime"
            train_X = X[(X["date"] >= train_start) & (X["date"] < train_end)]
            train_y = y[(y["date"] >= train_start) & (y["date"] < train_end)]
            test_X = X[(X["date"] >= test_start) & (X["date"] < test_end)]
            test_y = y[(y["date"] >= test_start) & (y["date"] < test_end)]
        else:
            raise ValueError("Time Column Not Found!")
        
        return train_X, train_y, test_X, test_y


    def _fit_predict(self) -> pd.Series | pd.DataFrame:
        """Fit and predict"""
        train_X, test_X, train_y, test_y = self._split_train_test(self.X, self.y, train_start="2010-01-01", train_end="2020-01-01", test_start="2020-01-01",)
        
        # temporarily save the index for backtest later
        train_idx = train_X[[self.time_col, 'product']]
        test_idx = test_X[[self.time_col, 'product']]
        
        train_X = train_X.drop('product', axis=1).set_index(self.time_col)
        train_y = train_y.drop('product', axis=1).set_index(self.time_col)
        
        model = self._model()
        
        model.fit(train_X, train_y, **self.model_params)
        pred = model.predict(test_X)
        
        # convert to DataFrame
        pred_df = pd.DataFrame(pred, columns=["pred"])
        pred_df = pd.concat(
            [pred_df, test_idx.reset_index(drop=True)], axis=1, ignore_index=True
        )
        
        return pred_df
    
    def rolling_fit_predict(self):
        """Rolling window fit and predict"""
        res = []
        
            
        return 
            


        



class PrepareXY:
    def __init__(
        self,
        alpha_dict: dict,
        target: pd.DataFrame,
        norm_X: callable,
        align_method: str = "inner",
        **kwargs,
    ) -> None:
        self.alpha_dict = alpha_dict
        self.target = target  # 已经shift过后的target
        self.norm_X = norm_X
        self.norm_params = kwargs.get("norm_params", {})
        self.align_method = align_method

    def alpha_to_X(self) -> pd.DataFrame:
        """Convert my alpha_dict to dataframe for ML training

        Returns:
            pd.DataFrame: columns=[factor], index=datetime
        """
        alpha_list = []
        factor_names = []
        for factor_name, factor in self.alpha_dict.items():
            factor = self.norm_X(factor, **self.norm_params)
            factor = factor.stack(dropna=False)
            alpha_list.append(factor)
            factor_names.append(factor_name)
        X = pd.concat(alpha_list, axis=1, keys=factor_names)
        X.index.names = ["date", "product"]
        return X

    def target_to_Y(self) -> pd.Series:
        """Convert my target to pd.series for ML training

        Returns:
            pd.Series: index=[date/datetime, tp/product]
        """
        y = self.target.stack(dropna=False)
        y.name = "target"
        y.index.names = ["date", "product"]
        return y

    def align_X_Y(self) -> Union[pd.DataFrame, pd.Series]:
        X = self.alpha_to_X()
        Y = self.target_to_Y()
        Y, X = Y.align(X, join=self.align_method)
        return X, Y

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.align_X_Y()
