import pandas as pd
import numpy as np


def test_shift_and_drop():
    df = pd.DataFrame({'Adj Close': [10, 20, 30, 40, 50]})
    future_days = 2
    df['Prediction'] = df['Adj Close'].shift(-future_days)

    expected_prediction = pd.Series([30, 40, 50, np.nan, np.nan], name='Prediction')
    pd.testing.assert_series_equal(df['Prediction'], expected_prediction)

    X = df.drop(['Prediction'], axis=1)
    X = X[:-future_days]

    assert list(X.columns) == ['Adj Close']
    assert X.shape == (3, 1)

    y = df['Prediction'][:-future_days]
    expected_y = pd.Series([30.0, 40.0, 50.0], name='Prediction')
    pd.testing.assert_series_equal(y.reset_index(drop=True), expected_y)
