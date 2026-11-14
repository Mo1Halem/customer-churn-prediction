# Customer Churn Prediction

**Live demo:** https://customer-churn-prediction-c3k6tzdacawk9fxsbmy92y.streamlit.app/

The app may take a few seconds to wake up if it has been idle.

I built this on the Telco Customer Churn dataset (7,043 customers, about 26% of them churned) to see how far a simple, honest pipeline can go, and to turn the predictions into something a retention team could actually use.

Everything is in `main.ipynb`. `predict.py` scores customers from the command line, and `app.py` is the Streamlit app behind the live demo.

## The problem

When a telecom customer cancels, the company loses recurring revenue, and keeping a customer is usually cheaper than winning a new one. The goal is to flag customers who are likely to cancel so the retention team knows who to contact, and to understand what sets them apart.

The target is `Churn` (yes/no). Contacting a customer costs money, so plain accuracy isn't the right yardstick. I look at recall and ROC-AUC for the model, and at the net benefit of a retention campaign for the decision.

The data is IBM's Telco Customer Churn sample dataset (also on Kaggle): 7,043 customers and 21 columns. The only missing values are 11 blank `TotalCharges` entries, all for customers with zero months of tenure.

## What I found

The model ranks customers well but not perfectly. On the 1,409 customers I held out as a test set:

| Metric | Value |
|---|---|
| ROC-AUC | 0.845 |
| Churn recall | 0.71 |
| Churn precision | 0.56 |
| F1 (churn class) | 0.63 |

I compared four models with 5-fold cross-validation on the training data. Logistic Regression (AUC 0.848) and Gradient Boosting (0.847) finished level, Random Forest was behind at 0.823, and a single Decision Tree at 0.660 was clearly the weakest. Tuning Gradient Boosting got its CV score to 0.8505, but on the test set it still came out the same as Logistic Regression (0.846 vs 0.845). Since they perform the same, I kept Logistic Regression because its coefficients are easy to explain.

The risk groups behave the way they should: customers I labelled Low, Medium and High risk actually churned 6.9%, 28.1% and 56.0% of the time. The High group is a third of customers but contains 71% of everyone who left.

What drives churn, from permutation importance and the raw data:

- Tenure matters most. People who leave have a median of 10 months with the company, versus 38 for people who stay.
- Month-to-month contracts churn at 42.7%, against 11.3% for one-year and 2.8% for two-year contracts.
- Fiber optic customers churn at 41.9%, DSL customers at 19.0%.
- Paying by electronic check goes with 45.3% churn; other payment methods sit around 15-19%.

![Feature importance](images/Important%20Features.png)

![Calibration curve](images/Calibration%20Curve.png)

![Risk performance](images/Risk%20Performance.png)

## How it's built

All preprocessing (scaling, one-hot encoding, grouping tenure into buckets) sits inside a scikit-learn `Pipeline`, so nothing leaks from the test set into training. Models are compared with cross-validation, and the test set is only touched at the end.

I tried 15 extra features (counts of add-on services, charge ratios, a few interactions). None of them moved the score beyond noise, except grouping tenure into buckets, which helped Logistic Regression a little because the effect of tenure isn't a straight line. That is the only one I kept.

For the decision threshold I didn't stop at the default 0.5. I searched for the best cutoff with cross-validation on the training data, and checked it against the net benefit of a retention campaign: I assumed a retention offer costs 50, saves 30% of the churners it reaches, and a saved customer is worth 500. Both approaches pointed to 0.35, which is close to the textbook value for those numbers (50 / (0.30 x 500) = 0.33).

## Running it

```bash
pip install -r requirements.txt
```

The dataset is included in the repo, so the notebook runs from top to bottom as it is. Running it also regenerates `customer_churn_model.pkl`.

To try the web app locally:

```bash
streamlit run app.py
```

To score customers from the command line:

```bash
python predict.py                              # one built-in example customer
python predict.py customers.csv                # every row of a CSV
python predict.py customers.csv -o scored.csv  # save the output
```

The CSV needs the raw Telco columns. `customerID` is optional and `Churn` is ignored. Keep `churn_utils.py` in the same folder as `predict.py`, because the saved model looks for it by name. `requirements.txt` pins pandas, numpy and scikit-learn to the versions used for training, because the saved model depends on them.

## Things to keep in mind

The money figures (500, 50, 30%) are my own assumptions, not company data. Swapping in real numbers changes the best threshold but doesn't require retraining anything.

This dataset seems to top out around 0.85 AUC; every reasonable model I tried landed within a few thousandths of that. The difference in net benefit between Logistic Regression and Gradient Boosting (16,100 vs 16,400) is too small to call on 1,409 test customers.

Everything above is correlation. A contract discount for month-to-month customers is a reasonable idea, but only an A/B test would show whether it really reduces churn.

## Ideas for the business side

- Offer high-risk month-to-month customers a discount to move to an annual plan.
- Put extra effort into the first six months, when the churn rate is highest.
- Look into why Fiber optic customers leave at twice the DSL rate (price? quality? competition? the data can't say).
- Nudge electronic-check payers toward automatic payment.
- Send retention offers only to customers above the 0.35 probability cutoff.

## Files

```
main.ipynb                 analysis and modelling
app.py                     Streamlit web app
predict.py                 scoring script (also used by the app)
churn_utils.py             tenure-bucket function used by the pipeline
customer_churn_model.pkl   saved pipeline, threshold and expected columns
requirements.txt
WA_Fn-UseC_-Telco-Customer-Churn.csv   dataset
```
