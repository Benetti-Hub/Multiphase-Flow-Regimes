# Two-Phase Adiabatic Flow Pattern Estimator

(https://flowpatternpredictor-szgzoob54q-oa.a.run.app)

The present work is dedicated to the determination of the most promising machine learning algorithm and the best set of dimensionless features able to solve the challenging classification problem of adiabatic flow pattern prediction. Thus, using an heterogeneous dataset retrieved from literature, a machine learning pipeline which considers physical quantities as input and flow regimes as targets is developed and optimized. In the first step of the pipeline, the data used for this thesis is pre-processed by replacing missing or incorrect values and by inserting physical quantities of interest. Then, we benchmark various machine learning models against each other and select the best suited one to solve the problem at hand. In the interest of time, we limit our search to state-of-the-art algorthms explored by AutoGluon, the AutoML framework developed by Amazon Web Services. Using the best algorithm identified in the previous step, a wrapping feature selection algorithm is applied, and the best combination of dimensionless features is determined. Following, we repeat the model selection step, this time by using only the selected feature subset, in order to identity the best machine learning algorithm (and its corresponding hyperparameters) resulting in the highest cross-validation accuracy. Said model only employs five dimensionless features, but it is able to achieve a test accuracy of 95.9\% and a macro averaged F1-score of 95.4\%, greater than any other model currently available in literature. Proper model evaluation is also carried out using the proposed pipeline, and its outputs show great accordance with some of the most well-known multiphase flow maps. The optimized machine learning model, as well as the link to its related web-app are available on GitHub.
