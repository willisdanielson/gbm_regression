# gbm_regression
Here I am using XGBoost to predict the total amount spent by customers based on socioeconomic factors and shopper behavior. 
This project includes exploratory data analysis, data cleaning, the final analysis, and a rough evaluation of the model.

At the moment, the biggest problem of this model is the heteroskedasticy in the resulting predictions. The model does very well when then
total amount spent is less than about 500, but as the amount grows the predictions start to wander from their true values. With the model
appearing to underestimate larger spending amounts.  

I think this is related to how skewed the target data is. In the future I want to build a separate GBM that has the task of binary 
classification, and train it to predict if a customer will be in the high or low spending brackets (Which is a task I think a GBM would
shine on). Then evaluate customers using two differently trained XGboost regressors.

The data are pulled from https://www.kaggle.com/imakash3011/customer-personality-analysis.

