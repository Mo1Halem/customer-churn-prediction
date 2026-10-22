# Customer Churn Prediction

I built this on the Telco Customer Churn dataset (7,043 customers, about 26% of them churned) to see how far a simple, honest pipeline can go, and to turn the predictions into something a retention team could actually use.

Everything is in `main.ipynb`. `predict.py` scores new customers with the saved model.

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

## How it's built

All preprocessing (scaling, one-hot encoding, grouping tenure into buckets) sits inside a scikit-learn `Pipeline`, so nothing leaks from the test set into training. Models are compared with cross-validation, and the test set is only touched at the end.

I tried 15 extra features (counts of add-on services, charge ratios, a few interactions). None of them moved the score beyond noise, except grouping tenure into buckets, which helped Logistic Regression a little because the effect of tenure isn't a straight line. That is the only one I kept.

For the decision threshold I didn't just maximise F1. I assumed a retention offer costs 50, saves 30% of the churners it reaches, and a saved customer is worth 500, then picked the threshold with the highest net benefit. Cross-validation chose 0.35, which is close to the textbook value for those numbers (50 / (0.30 x 500) = 0.33). On the test set this beats the default 0.5 cutoff.

## Running it

```bash
pip install -r requirements.txt
```

Run the notebook top to bottom (put the dataset CSV next to it). That also regenerates `customer_churn_model.pkl`.

To score customers:

```bash
python predict.py                              # one built-in example customer
python predict.py customers.csv                # every row of a CSV
python predict.py customers.csv -o scored.csv  # save the output
```

The CSV needs the raw Telco columns. `customerID` is optional and `Churn` is ignored. Two things to know: keep `churn_utils.py` in the same folder, because the saved model looks for it by name, and load the model with the same scikit-learn version it was trained with (check with `pip show scikit-learn`).

## Things to keep in mind

The money figures (500, 50, 30%) are my own assumptions, not company data. Swapping in real numbers changes the best threshold but doesn't require retraining anything.

This dataset seems to top out around 0.85 AUC; every reasonable model I tried landed within a few thousandths of that. The difference in net benefit between Logistic Regression and Gradient Boosting (16,100 vs 16,400) is too small to call on 1,409 test customers.

Everything above is correlation. A contract discount for month-to-month customers is a reasonable idea, but only an A/B test would show whether it really reduces churn.

## Ideas for the business side

- Offer high-risk month-to-month customers a discount to move to an annual plan.
- Put extra effort into the first six months, when most customers who leave do so.
- Look into why Fiber optic customers leave at twice the DSL rate (price? quality? competition? the data can't say).
- Nudge electronic-check payers toward automatic payment.
- Send retention offers only to customers above the 0.35 probability cutoff.

## Files

```
main.ipynb                          analysis and modelling
WA_Fn-UseC_-Telco-Customer-Churn.csv
churn_utils.py                      tenure-bucket function used by the pipeline
predict.py                          scoring script
customer_churn_model.pkl            saved pipeline, threshold and expected columns
requirements.txt
```