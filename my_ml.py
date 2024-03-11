"""
Author: Jet Deng
Date: 2024-02-21 10:08:15
LastEditTime: 2024-03-01 16:35:32
Description: ML工具类
"""

import pandas as pd
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
        train_method: str = None,
        **kwargs,
    ):
        self.X = X
        self.y = y
        self.model = model
        self.train_method = train_method

        if self.train_method != None:
            self.retrain_freq = kwargs.get(
                "retrain_freq", "6M"
            )  # Rolling/Expanding Window
            self.rolling_window = kwargs.get(
                "rolling_window", 10
            )  # Rolling Window Size default 10 years
            self.end_date = kwargs.get(
                "end_date", "2023-07-01"
            )  # Rolling Window End Date

        self.shift_num = kwargs.get(
            "shift_num", 10
        )  # Time Series Shift. Need to cut the first shift_num rows from the test data.
        self.train_test_size = kwargs.get("train_test_size", 0.8)
        self.model_params = kwargs.get("model_params", {})

    def fit_predict(self):
        if self.train_method == None:
            return self._fit_predict()
        elif self.train_method == "rolling":
            return self._rolling_fit_predict()
        elif self.train_method == "expanding":
            return self._expanding_fit_predict()
        else:
            raise ValueError("Train Method Not Supported!")

    def _split_train_test(self):
        """Split train and test data based on size"""
        # TimeSeriesSplit
        if isinstance(self.y.index, pd.DatetimeIndex):
            y_index = self.y.index.unique()
            train_split_date = y_index[int(len(y_index) * self.train_test_size)]
            test_split_date = y_index[
                int(len(y_index) * self.train_test_size) + self.shift_num
            ]
            train_X = self.X[self.X.index <= train_split_date]
            test_X = self.X[self.X.index > test_split_date]
            train_y = self.y[self.y.index <= train_split_date]
            test_y = self.y[self.y.index > test_split_date]

        # Non-TimeSeriesSplit
        else:
            train_X = self.X.iloc[: int(len(self.X) * self.train_test_size)]
            test_X = self.X.iloc[int(len(self.X) * self.train_test_size) :]
            train_y = self.y.iloc[: int(len(self.y) * self.train_test_size)]
            test_y = self.y.iloc[int(len(self.y) * self.train_test_size) :]

        return train_X, test_X, train_y, test_y

    def _model(self):
        """Initialize model"""
        if isinstance(self.model, str):
            model = eval(self.model)(**self.model_params)
        else:
            model = self.model
        return model

    def _fit_predict(self) -> pd.Series | pd.DataFrame:
        """Fit and predict"""
        train_X, test_X, train_y, test_y = self._split_train_test()
        model = self._model()
        if self.train_method == None:
            try:
                model.fit(train_X, train_y, **self.model_params)
                pred = model.predict(test_X)
            except Exception as e:
                print(f"Model fit predict failed: {e}")
        return pred

    def _rolling_fit_predict(self) -> dict:
        """Rolling fit and predict"""
        train_X, test_X, train_y, test_y = self._split_train_test()

        start_idx = test_y.index[0]
        split_dates = pd.date_range(
            start_idx, self.end_date, freq=self.retrain_freq
        )  # 保存划分日期

        # Rolling Fit and Predict
        res = {}  # Save the pred result for each split date
        for i in range(len(split_dates) - 1):
            model = self._model()
            train_end_date = split_dates[i]
            train_X_ = train_X[train_X.index <= train_end_date]
            train_y_ = train_y[train_y.index <= train_end_date]

            test_start_date = train_end_date + pd.Timedelta(
                self.shift_num / 5 * 7, unit="D"
            )  # Convert the shift_num to trading days
            if i == len(split_dates) - 2:
                test_end_date = self.end_date
            else:
                test_end_date = split_dates[i + 1]
            test_X_ = test_X[
                (test_X.index > test_start_date) & (test_X.index <= test_end_date)
            ]
            test_y_ = test_y[
                (test_y.index > test_start_date) & ((test_X.index <= test_end_date))
            ]
            try:
                model.fit(train_X_, train_y_, **self.model_params)
                pred = model.predict(test_X_)
                res[test_start_date] = pred
            except Exception as e:
                print(f"Model rolling fit predict failed: {e}")
        return res

    def _expanding_fit_predict(self) -> dict:
        """Expanding fit and predict"""
        train_X, test_X, train_y, test_y = self._split_train_test()

        start_idx = test_y.index[0]
        split_dates = pd.date_range(start_idx, self.end_date, freq=self.retrain_freq)


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
        self.target = target
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
        return X

    def target_to_Y(self) -> pd.Series:
        """Convert my target to pd.series for ML training

        Returns:
            pd.Series: index=[date/datetime, tp/product]
        """
        return self.target.stack(dropna=False)

    def align_X_Y(self) -> Union[pd.DataFrame, pd.Series]:
        X = self.alpha_to_X()
        Y = self.target_to_Y()
        Y, X = Y.align(X, join=self.align_method)
        return X, Y

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.align_X_Y()
