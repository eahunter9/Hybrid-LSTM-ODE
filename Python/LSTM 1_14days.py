
# fix python path if working locally
##from utils import fix_pythonpath_if_working_locally

#fix_pythonpath_if_working_locally()
#%load_ext autoreload
#%autoreload 2
#%matplotlib inline

# use darts plotting style
from darts import set_option

set_option("plotting.use_darts_style", True)
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts.dataprocessing.transformers import Scaler

from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import (
    InvertibleMapper,
    Mapper,
    MissingValuesFiller,
    Scaler,
)
from darts.utils.timeseries_generation import linear_timeseries
from darts import TimeSeries
from darts import concatenate

from darts.datasets import AirPassengersDataset, SunspotsDataset
from darts.metrics import mape
from darts.models import BlockRNNModel, ExponentialSmoothing, RNNModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models.torch import QuantileRegression

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

dataset = pd.read_csv("C:/Users/Elizabeth/OneDrive - National University of Ireland, Galway/calibration/waningimmunity/delay/behaviour/coviddeaths_1styear.csv")


data_covid = dataset[["Newdeaths"]][0:395]

data_covid = TimeSeries.from_dataframe(data_covid)


train = data_covid[0:300]
val = data_covid[301:329]

# Normalize the time series (note: we avoid fitting the transformer on the validation set)
transformer = Scaler()
##train_transformed = transformer.fit_transform(train[["SUM_no_new_admissions_covid19_p"]])
#val_transformed = transformer.transform(val[["SUM_no_new_admissions_covid19_p"]])
#series_transformed = transformer.transform(data_hp[["SUM_no_new_admissions_covid19_p"]])
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(data_covid[0:395])


my_model = BlockRNNModel(
    input_chunk_length=5,#5
    output_chunk_length=14,
    model="LSTM",
    hidden_dim=20, #20
    dropout=0,
    n_rnn_layers=3,#3
    batch_size=35,#35
    n_epochs=400, #400
    model_name="test_RNN",
    log_tensorboard=True,
    random_state=42,
    force_reset=True,
    save_checkpoints=True,
    likelihood=QuantileRegression(quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]),
)



my_model.fit(
    train_transformed,
    val_series=val_transformed,
    verbose=True#,
# past_covariates =  past_covariates_train,
 #val_past_covariates = past_covariates_val,

)


def eval_model(model):
    pred_series = model.predict(n=14, num_samples=500)#,past_covariates = past_covariates_train)
    backtest = my_model.historical_forecasts(series=train_transformed, 
                                            #past_covariates =  past_covariates_train,
  
                                          start=0, 
                                          retrain=False,
                                          verbose=True, 
                                          forecast_horizon=14)
    plt.figure(figsize=(8, 5))
   # series_transformed.plot(label="actual")
    data_covid[0:395].plot(label="actual")
    untransformed_series = transformer.inverse_transform(pred_series)
    #pred_series.plot(label="forecast")
    untransformed_series.plot(label="forecast")
    untransformed_backtest = transformer.inverse_transform(backtest)
    untransformed_backtest.plot(label='backtest')
    #backtest.plot(label='backtest')
    #plt.title("LSTM 1: 14 Day Prediction Period")#f"MAPE: {mape(pred_series, val_transformed):.2f}%")
    plt.title(f"MAPE: {mape(pred_series, val_transformed):.2f}%")

    plt.legend()

eval_model(my_model)




pred_series = my_model.predict(n=14, num_samples=500)
pred_lower = pred_series.quantile(0.025)
pred_lower = transformer.inverse_transform(pred_lower)
#pred_lower.to_csv("C:/Users/Elizabeth/OneDrive - National University of Ireland, Galway/calibration/waningimmunity/delay/behaviour/behaviour14block_lower.csv")

pred_upper = pred_series.quantile(0.975)
pred_upper = transformer.inverse_transform(pred_upper)
#pred_upper.to_csv("C:/Users/Elizabeth/OneDrive - National University of Ireland, Galway/calibration/waningimmunity/delay/behaviour/behaviour14block_upper.csv")

pred_median = pred_series.quantile(0.5)
pred_median = transformer.inverse_transform(pred_median)
#pred_median.to_csv("C:/Users/Elizabeth/OneDrive - National University of Ireland, Galway/calibration/waningimmunity/delay/behaviour/behaviour14block_median.csv")
