## Basic hypothesis (model):

$$ h_\theta(x) = \theta_o x_0 + \theta_1 x_1 + \theta_2 x_2....\theta_n x_n  $$
where $\theta_n$ is a parameter value to be calculated and $x_n$ is the value for the feature in the sample (training) data (with n features)

Basic cost function:
Using the above hypothesis, the cost is the distance between the y value in the training data and the y value that's calculated by the hypothesis $h_\theta(x)$.

The idea is that the hypothesis/function you're tweaking by adjusting ${\theta}$ values will produce a result that matches very closely all training data y values and *also* predicts y values for unseen training data accurately.

So the cost function tells you how far off you are.  Obviously a cost of 0 means that your prediction exactly matches the y value in the training data for the row of feature data.

$$ J(\theta) = \frac{1}{2m} * \sum_{i=1}^m(h_\theta(x_i) - y_i)^2$$

That's mean squared error for all training data: averaged distance from y for all predictions in your hypothesis with a given set of $\theta$ values.  The $\frac{1}{2}$ in front is just to make the math a bit easier when you start taking the derivative of that function.  The $\frac{1}{m}$ is to get the mean since m is the number of training examples.


## Gradient Descent:

This is an iterative method to figure out the values for $\theta$ that minimize the cost function.  Once you figure that out, you have your working hypothesis into which you can plug unseen data and (hopefully) get a good prediction.

First off, you'll need to have a $\theta$ value for each feature so you can use the same formula on each feature while calculating gradient descent.  Without a $\theta$ for the zeroth feature (also called the intercept since in the y=mx + b basic formula for a line, the y is the first feature), you'd have to treat $\theta_0$ differently.  Think of $\theta_0$ as the base value.  So, in the housing prices example, the base value is a starting house price without taking into account any features.  So, the convention is just to multiply this first feature by 1.  You just need to remember to add that to your training data as the first column - all ones.

Gradient descent uses partial derivatives with respect to each $\theta$ value, one at a time.  You calculate the cost iteratively while tweaking $\theta$ values in the direction of a lower slope, incrementally moving toward the minimum value for the cost.  If you think of the cost function's graph as a parabolic shape on a graph (bowl shaped, convex), you're looking for the bottom of that curve.

for each iteration, calculate new $\theta$ values:

$$    \theta_0 = \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{i}) - y^{i})*x_0^{(i)}) $$
$$    \theta_1 = \theta_1 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{i}) - y^{i})*x_1^{(i)}) $$
...

$$    \theta_n = \theta_n - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{i}) - y^{i})*x_n^{(i)}) $$


That $\alpha$ is the "learning rate" the value to multiply the slope of the tangent by as it gets smaller and you step towards the minimum.  Too large and gradient descent may never converge; too small and it'll take a long time.  Note how, since the slope gets smaller, the increment by which you're reducing $\theta$ also gets smaller.  How many times to calculate (the number of iterations) needs to be chosen carefully.  You can graph the result of the cost function during these iterations to make sure it's converging towards the minimum at a reasonable rate, and not increasing.


One importing thing here: simultaneous update.  You need to calculate each new theta value with the existing set *before* updating all.  You don't, for example, want to use a newly calculated $\theta_0$ while calculating the $\theta_1$ on the same iteration.

This can be vectorized to avoid the repetitive calculation for each new $\theta$ shown above:

$$ \theta = \theta - \alpha\frac{1}{m}((X\theta-y).*X)$$

That's dot-product multiplication at the end. (That's the version that worked for me in homework. Not honestly sure how...) It's worth calling out that $\theta$ here is a column vector (one column of values).  That way $X\theta$ means that features from X line up with the $\theta$s for the multiplication and the result is a column vector of m rows 1 column.  y here is also a column vector (m rows, 1 column) so the subsequent subtraction is element-by (remember that y was produced by just lopping off the last column of the training data)   Ng suggests later during logistic regression lectures that a vectorized version of this is:

$$ \theta = \theta - \alpha\frac{1}{m}X^T(X\theta-y)$$


## Feature scaling

Ideally features are all within the same basic range of values, somewhere between -1 and 1 or close to that but that never really happens naturally.  Gradient descent works better if they are.  You can scale them to be so like this:

$$  new feature value f = \frac{f - \mu}{\sigma}  $$

where $\mu$ is the mean feature value over all samples and $\sigma$ is the standard deviation.  Octave (and I'm assuming numpy) have functions to calculate the mean and sigma over all sample rows for each column.  So you can use those and perform vector subtraction and dot-quotient operations to implement that formula efficiently (see homework number 1 for Ng's course)

## Logistic Regression

This is the technique used for binary classification problems - spam or not, success or failure, malignant or benign.  So the predictions are not continuous values but discrete: a 1 or a 0.  So, our hypothesis is the same but the cost function is no longer linear.  All the values should hew toward zero or 1 except for within a relatively small "decision boundary" range where they are less definitively one or the other.  So we're going to use a sigmoid function to help us map hypothesis values to one of those values.   

The sigmoid function asymptotes to 0 or 1.  At 0, the value is exactly 0.5.  This is a sigmoid function:

$$ \frac{1}{1+e^{-z}} $$

The values of the function that are greater than 0.5 are positive cases, less then .5, negative ones.  One important insight about this function is that when z = 0, this function returns exactly 0.5.  When z > 0, y is > 0.5. When z < 0, y < 0.5.  We're just going to wrap our hypothesis in this function by plugging it in as z:

$$ \frac{1}{1+e^{-h_\theta(x)}}  $$

Or, the way Ng defines it, call this a parameterized function z that takes our hypothesis as a parameter:

$$ z = g(X\theta) = \frac{1}{1+e^{-X\theta}}$$

Worth noting here that $X\theta$ will result in a vector.  So how do you take e to a vector power?  In the homework, Ng instructs that the sigmoid function we write should take the sigmoid value for all elements independently.  So, this function will also return a vector of values all run through the sigmoid function.  Guess this makes sense in that we want a value for each $\theta_jX_j$ values.  But does it?  We're mapping the entire hypothesis with all thetas to the sigmoid funcion.  You need to think about logistic regression differently. The hypothesis for logistic regression doesn't define a line that attempts to approach all datapoints, like we did for linear regression but instead just predicts 0 or 1.  If you think of the graph of all points for a logistic regression problem, those points don't have anything to do with the decision boundary.  The decision boundary is where the hypothesis (wrapped $X\theta$) is equal to 0.5 or $X\theta$ is equal to 0.  So, given the hypothesis, how do you graph the decision boundary instead which is *not* the same as the hypothesis result of 0 or 1?  That's given in the homework but I can't figure it out.  Assuming a 2 feature dataset (with 1s added in the first column) the x axis just needs to span the entire range of x values for all datapoints.

## Logistic Cost Function

The cost function for logistic regression (classification) needs to be different from that used for linear regression because if we used the original cost function, the sigmoid function would cause the cost to be wavy with more than one minimum (local minima).  We need to write the cost function differently to insure that it's convex (one minimum) when the hypothesis is wrapped in the sigmoid function as described above.

Turns out, log functions can be used because they can have the effect of asymptoting toward 0 or 1, depending on whether y is 0 or 1.  So it's really different cost functions for each of those positive or negative cases but they can be combined into this:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)}log(h_\theta(x)) + (1 - y^{(i)})log(1-h_\theta(x^{(i)}))]$$


Notice what happens to this function when y = 0 or y = 1.  One of the terms disappears.  Also, keep in mind that when figuring out cost, we're really figuring out how far the hypothesis is from predicting the correct value.  So the asymptoting of the log functions works so that we penalize a 0 prediction when 1 is correct very heavily and vice versa.  If we just took the difference between the prediction and correct answer as cost, cost would always just be 1 for a wrong prediction and gradient descent would probably have trouble walking down the cost curve (my understanding)

Remembering that our hypothesis (h, here) is the usual ($X\theta$) wrapped by the sigmoid function (see above), a vectorized version of the logistic cost function is:

$$ J(\theta) = \frac{1}{m}(-y^Tlog(h) - (1-y)^Tlog(1-h))$$

## Gradient descent for logistic regression

Take the derivative of this and, again, through the magic of math, the gradient descent function is exactly the same as before.  A vectorized version (see above for more info):

$$ \theta = \theta - \alpha\frac{1}{m}X^T(X\theta-y)$$

So, for logistic regression, the hypothesis is wrapped in a sigmoid function so that values hew toward zero or one and the cost function uses log functions which asymptote toward zero or one.  But the partial derivative of that for gradient descent is the same as that used for linear regression.

## Other options besides gradient descent to optimize $\theta$ values.

BFGS, L-BFGS - provided by Octave and probably also numpy.  Those take a cost function and theta vector as parameters along with some other settings - like the number of iterations.

## Multi-class classification

If you need to group into multiple classes, you could just write separate hypotheses (classifiers) to do a "one vs. all" approach.  Run all data through each of these and pick the class that has the highest result (probability) for each sample.  If you want classify into a number of different classes, you might also want to consider k-means which, I believe, does the same thing...

## Overfitting / Regularization

As you train your algorithm/hypothesis to be able to predict, there's a risk that you optimize so well for the training set that your algorithm performs poorly on unseen data.  That's called overfitting.  And it sometimes results in a very wavy function that accommodates all points in your training set.  In order to be generally applicable, though, you want the function to be relatively smooth.  The more features, or the more complex the hypothesis function, the more likely you'll overfit the data.  This applies to both linear and logistic regression.  In order to avoid, you could use fewer features or use a model selection algorithm (see later).  Or, you could use regularization which reduces the magnitude of certain features so that they have less influence over the function's shape - good when you have a lot of slightly useful features.

To do this, you modify the cost function, appending terms that multiply certain features by a factor that increases their cost.  With that factor/multiplier, the thetas for those terms would need to be smaller to reduce overall cost - thus reducing those terms' influence.  Here's a sample adjusted cost function with regularized terms:

$$ J(\theta) = \frac{1}{2m} * \sum_{i=1}^m(h_\theta(x_i) - y_i)^2 + 1000 * \theta_3^2 + 1000 * \theta_4^2$$

In other words, square the third and fourth theta and multiply them by 1000 thereby increasing the cost of those features and causing those thetas to be shrunk when finding the global minimum via gradient descent (or potentially other algorithms that use a cost function).  Not sure how you pick the factor or why you square them either - maybe the squaring is to make sure they're positive values when adding cost to them...

And you could just do this to all thetas:

$$ J(\theta) = \frac{1}{2m} * \sum_{i=1}^m(h_\theta(x_i) - y_i)^2 + \lambda \sum_{j=1}^n\theta_j^2$$

(remember n is the feature index, not the row index and j starts at 1 so that you don't muck with $\theta_0)$  I'm unclear on why you'd do this to all features.  If you're trying to smooth out some higher order terms that are cubed or to the fourth power, to make a function "more quadratic", why would you also touch other terms that may just be squared?

Also, $\lambda$ here is the "regularization parameter".  If you chose a $\lambda$ that's too large, you could underfit.  Consider if all terms but the first approach zero.  You'll get a line.


## Linear regression gradient descent with regularization

Since our cost function changes with regularization, so does the partial derivative.  Since we don't want to touch $\theta_0$, gradient descent $\theta_0$ calculation doesn't change but is still included in each iteration.

Repeat (for $\theta_0$):

$$    \theta_0 = \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{i}) - y^{i})*x_0^{(i)}) $$

And then for all j > 0:

$$    \theta_j = \theta_j - \alpha[\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{i}) - y^{i})*x_1^{(i)}) + \frac{\lambda}{m}\theta_j]$$ 

for each iteration.

For that second equation, it can be re-written so that the second term is exactly the same as before which might make the coding easier...:

$$    \theta_j = \theta_j(1 - \alpha) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{i}) - y^{i})*x_1^{(i)})$$ 



## Normal equation with regularization

Missing from above, the "normal" equation that you can use instead of gradient descent to calculate thetas and requires no feature scaling.  But you can't use it when feature count (n) is greater than sample data (m).

$$ \theta = (X^TX)^{-1}*(X^Ty)$$

With regularization, that becomes:

$$ \theta = (X^TX + \lambda*L)^{-1}*(X^Ty) $$

where L is an identity matrix with dimensions (n+1)x(n+1) (number of features + 1 in both directions) AND position [0,0] set to 0 instead of 1 ([1,1] in Octave since it's stupidly 1 based). (eye(n) in Octave then update [1,1]).


## Logistic regression with regularization

The cost function needs to be adjusted with regularization parameters:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)}log(h_\theta(x)) + (1 - y^{(i)})log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$$

And gradient descent is the same as for linear regression with regularization above:

Repeat:

$$    \theta_0 = \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{i}) - y^{i})*x_0^{(i)}) $$

And then for all j > 0:

$$    \theta_j = \theta_j - \alpha[\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{i}) - y^{i})*x_1^{(i)}) + \frac{\lambda}{m}\theta_j]$$ 

for each iteration.


## Great cheat sheet for all regression, gradient descent, regularization formulas:
https://medium.com/ml-ai-study-group/vectorized-implementation-of-cost-functions-and-gradient-vectors-linear-regression-and-logistic-31c17bca9181













