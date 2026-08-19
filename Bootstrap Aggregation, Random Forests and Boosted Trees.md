## The Bootstrap
__Bootstrapping__ is a statistical resampling technique that involves random sampling of a dataset _with replacement_. It is often used as a means of quantifying the uncertainty associated with a machine learning model.

- The idea is to repeatedly sample data with replacement from the original training set in order to produce multiple seperate training sets. These are then used to allow "meta-learner" or "ensemble" methods to _reduce the variance of their predicitons_, thus greatly improving their predictive performance.
- Two of the following ensemble techniques - bagging and random forests - make heavy use of bootstrapping techniques, and they will now be discussed

## Bootstrap Aggregation
Main drawbacks of DTs is that they suffer from being _high-variance estimators_. This means that the addition of a small number of extra training observations can dramatically alter the prediction performance a learned tree, despite the training data not changin to any great extent.

One way to mitigate against this problem is to utilise a concept known as bootstrap aggregation or __bagging__.
The idea is to combine multiple learners (such as DTs), which are all fitted on sperate bootstrapped samples and average their predictions in order to reduce the overall variance of these predictions.

>[!Reminder]
>Why does this work? James et al (2013)[[2]](https://www.quantstart.com/articles/bootstrap-aggregation-random-forests-and-boosted-trees/#ref-isl) point out that if  independent and identically distributed (iid) observations  are given, each with a variance of  then the variance of the mean of the observations,  is . That is, if the average of these observations is taken the variance is reduced by factor equal to the number of observations.

If B separate bootstrapped samples of the training set are created, with seperate model estimators $\hat{f}^b ({\bf x})$, then averaging these leads to a low-variance estimator model, $\hat{f}_{\text{avg}}$:

$$\begin{eqnarray}
\hat{f}_{\text{avg}} ({\bf x}) = \frac{1}{B} \sum^{B}_{b=1} \hat{f}^b ({\bf x})
\end{eqnarray}$$
This procedure is known as __bagging__. It is highly applicable to DTs because they are high-variance estimators and this provides one mechanism __to reduce the variance substantially.__

Carrying out bagging for DTs is straightforward. Hundreds or thousands of deeply-grown (non-pruned) trees are created across B bootstrapped samples.
