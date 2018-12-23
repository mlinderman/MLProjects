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

for each iteration.  My vectorized version looks like this (see homework for costFunctionReg (ex2 or ex2 lrCostFunction)):

           grad = 1/m * (transpose(X) * (h - y)) + lambda/m * [0; theta(2:end)];

(That's not doing gradient descent (it's missing the theta - alpha part in the beginning), but just producing the gradient for each theta).  To walk through this, the result you're seeking is a vector of gradients for each theta (think about the non-vectorized version with a loop that calculates a gradient for each theta, adding sum of lambda * thetas to only the non-zero theta gradients). So just to check, X is matrix of samples, h is column vector of predictions, y is column of answers (1 per sample), m x 1. So h-y is element-by operation and result is a single column vector, m x 1, transpose(X) is an n x m matrix.  That multiplied by h-y is then a n x 1 matrix, 1 row per theta, and that's what you want!!! The result is a vecttor of gradients, 1 per theta (that are subtracted from the corresponding theta during gradient descent - but here we're just producing the gradients). Adding the regularization parameter changes nothing about the dimensions of the vector.



## Great cheat sheet for all regression, gradient descent, regularization formulas:
https://medium.com/ml-ai-study-group/vectorized-implementation-of-cost-functions-and-gradient-vectors-linear-regression-and-logistic-31c17bca9181



## How to decide on the number and type of features

I include this here not because it's a part of Ng's course but because it keeps coming up as a question in my mind: how to decide on whether to multiply base features together to come up with another.  In the neural networks section of the course, Ng talked about the range of possible features when you have 100 base feature (~5000) when you square each base feature and also multiply each feature against other features.  Something like : $x_1^2, x_1x_2, x_1x_3, x_1x_4.....x_1x_100$  and then again for $x_2$:  $x_2^2, x_2x_3, x_2x_4, x_2x_5......x_2x_100$  and so on.  What I'm getting from that is that you may not know up front which to choose but that as you add more, you can better fit unusual test data shapes (and maybe overfit them - see above). 

## Neural networks
When you have a lot of base features, you can use up a lot of additional feature space while trying to fit any sort of unusual array of sample data.  Consider the logistic examples that Ng used as demonstrations about regularization/over-fitting.  If the best way to fit these things well is by adding features, you might need to work with lots of features - in the thousands.  Think of the problem of computer vision where you may be comparing features of images that contain thousands of pixel values - even if they're only grayscale images, if you consider each pixel a feature.  Neural networks are useful in cases like these where the feature set (n) is very large.

A single "neuron" can be thought of as a single hypothesis as in logistic regression where inputs are the features and theta values are sometimes called "weights".  The input nodes in a neural network (the leftmost layer) are like the features $x_n$, one input node per feature.  $\Theta^{(j)}$ (uppercase theta), then, is a matrix, of thetas that communicates theta values between layers j and j + 1.  And $a_i^{(j)}$ is the activation node i in layer j.  By the way, the first activation layer is one to the right of the input layer so it's actually the second layer and so called $a_i^{(2)}$ 


Neural net activation nodes always use a sigmoid activation function (just like with logistic regression in order to map to 0 or 1). Really?  Why couldn't activation functions be anything?   You're just creating features for the next layer.  So far you've only seen activation functions using sigmoid functions - is that all that's possible with neural nets?

So each activation layer is "learning" it's own set of features which are, in turn, passed on to the next activation layer.  Input layer (x's) * $\Theta^{(1)}$ gives you inputs to $a^2$ and so on.  The multiplication of the $X\Theta^n$ that occurs in every layer Ng has abbreviated to z.  So, in each layer there's a $z_n^j$ (not sure about the super and subscript letters there but the point is that there's a different set of x's and thetas for each activation node in a neural network).  After the first activation layer, the x's (inputs)  subsequent layers are the a's that result from each activation layer.

Had a tough time with exercise a3 getting my head around how to do the forward propogation math.  In the end, some good insights were that it really isn't too disimilar to a single logistic regression problem (assuming you're always doing logistic regressions - but even if you're not).  The hard part was understanding that the input layer is a vertical vector of X's.  And that all of those X's get sent to *each* activation node in a layer.  Each activation node though has it's own $\theta$ taken from rows of $\Theta$.  So that means $\Theta$ has as many rows as there are activation nodes in the layer.  You add a corresponding $x_0 = 1$ to each activation layer's x's.  I imagine that later there will be a single $\Theta$ rather than 2 distinct $\Theta$s like in exercise 3's 3 layer network.  A single $\Theta$ then would have a row dimension for every activation node in a layer, and a matrix of those rows for each activation layer.  So that's 3 dimensions.  But so far, in Ng's lectures, the $\Theta$s have all been distinct matrices for each layer on a neural net.

Another good insight: in the exercise, we started with 400 inputs (x's, 1 per each pixel in a 20x20 gray scale image) and each $\Theta^{(1)}$ row then also had 400 + 1 thetas (weights).  And there were some 5000 rows of training data.  So, a 5000 x 400 matrix.  (But be careful considering all sample rows at once before the network is trained.  It worked to do a single pass through the network with trained thetas on all rows to com up with predictions.  But to train the network using back-propogation (learned about later), you take one row of sample data at a time and do forward and then backward propogation, accumulating the deltas as you go. In the end,I discovered that it was okay that you did forward propogation for all samples first because each layer of outputs is a row for each sample and you can just iterate over those rows as you do back prop without having to calculate them at the same time.  Neither case, calculating the outputs for all samples at once before iterating over individual samples for back prop OR doing forward then backward prop for one sample at a time seems inherently to be more efficient.)  

Because the next layer only had 25 activation nodes, $\Theta^{(1)}$ had only 25 rows.  So that 25 x 400 $\Theta^{(1)}$ matrix has the result of taking your original input layer (5000 x 400) and reducing that down to 5000 x 25 and, in the output layer, by the same process, down to 5000 x 10.  Those are your predictions (classification into 10 classes).  So in the traditional diagram of a neural network, consider the input layer as the X matrix, turned on its side, but still having all its training rows.  I fell into the habit of thinking about that input layer as a single set of 400 values.  But, in terms of matrix math, I needed to think about all of the training data so you can align the features up with the thetas.  (After the network is trained, that wouldn't be training data.  It would be test or real data).  I had to draw some pictures of the matrices involved in each step.  I don't honestly think it matters how you orient the matrices as long as you're multplying features by thetas.  Transpose is your friend and so is printing out the dimensions of arrays for confirmation.  This got me where I needed to go, ultimately:

$$ a2 = sigmoid(\Theta^{(1)} * transpose(X)) $$

And then the same math to get to a3 though did have to transpose again at some point to be able to find the max for each of the 10 columns of numbers for each training example, round that and line up the predictions as 1-10 in a single column vector so that it could be compared to Y.

Some other insights:  remember that the $\Theta$'s are two-dimensional matrices that, in effect, map the number of output nodes in one layer to the number of input nodes in the next and have no reference to the number of rows in the training data.  In this particular case, there were 2 $\Theta$s, the first mapping 401 inputs to 25 outputs (so, a 26 x401 matrix) and the second mapping 26 inputs to 10 outputs (our predictions), a 10 x 26 matrix.  (rows in the $\Theta$ matrices correspond to number of outputs.)


## Cost function for logistic Neural networks

So, this is basically the same cost function we used for logistic regression (utilizing log functions) but it's taken further to abstract it out to apply to all activation nodes in a layer and all layers in a neural network.  So there's some extra summing to be done across nodes and layers. Our original cost function for logistic regression (with regularization) was:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)}log(h_\theta(x)) + (1 - y^{(i)})log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$$


Our revised version to accommodate all nodes, layers needs a bit more notation.  Here K is the number of output nodes, L is the layer, $s_l$ is the number of nodes in layer l.


$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K [y_k^{(i)}log(h_\theta(x)) + (1 - y_k^{(i)})log(1-h_\theta(x^{(i)})_k)] + \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=l}^{s_l} \sum_{j=1}^{s_l+1}  (\theta_{j,i}^{(l)})^2 $$


## Back Propogation in Neural Networks

As with linear and logistic regression, the goal is to minimize the cost by adjusting $\Theta$ values by way of gradient descent or another algorithm.  Gradient descent and those other algorithms require us to be able to find the partial derivates of the cost curve with respect to each theta, and then use simultaneous update to update all at once.  For neural networks, it's pretty much the same operation but multiplied by the number of nodes in the network, one hypothesis per node (and, therefore, one set of $\Theta$s per node).  Taken one step at a time we first need to compute the output of the neural net and compare it to y (for each training example) - that's the cost between the result of the network and the real values. But it's not for all thetas.  The deltas for the output layer are relatively simple to calculate - you just subtract the corresponding y values for the current sample. (I use the plural here because some networks have multiple output nodes - for categorization, for example, so you need to translate those y values into a vector that lines up with the categories.  Then you can subtract y elements from output node elements $a^L - y^{(i)}$. 



So, again, one step at a time, you first need to compute the result so do forward propagation to come up with that by doing:

1. multiply input values x $\Theta^1$ and apply sigmoid function to all values to come up with $a^2$ :  $a^2 = sigmoid(\Theta^1 * X)$, just making sure that the matrices align, as always, features to Thetas.
2. Add a bias input node (by adding a column to $a^2$), then do the same with $a^2$ and  $\Theta^2$ to come up with $a^3$, and so on until you arrive at the output layer.
3.  So now you need to figure out the "error" or delta ($\delta$) for each node (j), in each layer(l): $\delta_j^{(l)}$.  The easy one is the last "error" between the output of the neural net (each node, if multiple) and the corresponding y.  So, that's $a_j^{(4)} - y_j$ in a four layer network. Remember, the neural net is classifying into buckets, with hopefully one bucket having a value > .5 and the rest less.  So, to compare that to Y values, you need to change the $y^{(i)}$ values to be represented the same way (as a vector of 1's and 0's). 
4. Now you continue backing through the NN (back propogation) between layer L-1 and L-2 finding deltas successively using:

$$ \delta^{(l)} = (\Theta^{(l)})^T\delta^{(l+1)} .* (a^{(l)}) .* (1 - a^{(l)}) $$

What that's doing is multiplying the deltas for each output in the subsequent layer times the $\Theta$s in the current layer so that each Theta value gets the output delta "applied" to it.  So a delta for the output is influencing the thetas (and then, the deltas) in the previous layer.  The second part is the derivate of the output nodes for the current layer.  That gives you the deltas for each non-output layer.  And you only have to do this calculation back to a2 because a1 is the input layer.  You do update the first layers Thetas ($\Theta^1$) but you only need the next layers deltas to do that.  See below.

One other thing here.  You need to lop off the bias deltas from each layer's deltas before accumulating them on each iteration.

Once you have those delta vectors, you can get the gradient by which to update each $\theta$ in the next forward pass through the network. So, you iteratively add these to an "accumulator" as you process each row:

$$ \Delta^{(l)} = \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T  $$


You *don't* calculate a delta for L1 since those are the input.  But you then need to calculate the average deltas for each $\theta$ over all training rows that you've accumulated deltas for.  This obviously happens outside the loop for processing each row:

$$\Delta^{(l)} = \frac{1}{m} * \Delta^{(l)}$$


and that's your gradient, one per theta, all the way through the network.  Missing though is the regularization.  You can add that to each layer when you perform the average or afterwards.  (The assignment broke it up so I did them separately) To do them at the same time, the forumula above becomes (only for j > 0 because we don't regularize the bias unit):

$$ \Delta^{(l)} = \frac{1}{m} * \Delta^{(l)} + \lambda\Theta^{(l)} $$

Excluding $\Theta_0^{(l)}$ which is the bias unit for each layer.

Intuition: back propogation is sort of the same thing as forward propogation but we're replacing the inputs (X's in the input layer and a's thereafter) with deltas.  On the forward pass, thetas get multiplied by inputs.  In the backwards direction, they're multiplied by the deltas and we're assigning relative "blame" for the error by way of the derivatives. Those relative blames are the gradients, I think.

When I did the back propogation exercise in Ng's course, I had to transpose some things that he didn't transpose but, again, as long as you're aligning matrices by the right dimensions, it still works.  That may have been because he assumes some orientation to a matrix that's different from what I assumed.  The key is to try to visualize what you're doing.  I commented extensively in nnCostFunction (where backprop is performed) with other insights.


## Unrolling Theta vectors for use in advanced functions

When using fminunc or other octave (or Python) functions to calculate the optimal values for Theta for neural networks when you have multiple theta matrices, you'll need to "unroll" the vectors and pass them as one long vector to those functions.  To do that, in Octave, you can do this:  

    thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
    or
    deltaVector = [ D1(:); D2(:); D3(:) ]

basically, that says to string all rows, all columns out in succession.  (Though I'm confused by the semicolon as I'd think that would create a new row for each but what actually happens is that this creates a column vector so the semicolons make sense.)  Just note that this unrolls by columns so all the elements in column 1, then in column 2, etc.


## Gradient checking

Because neural networks are hard and you can easily create a buggy network that looks like it's working, there's a way to estimate the derivatives of the thetas and compare those to what your backprop calculations come up with.  Ng says this is a way to eliminate all problems and that he uses it every time.  If you imagine a single theta and a graph of the function J($\Theta$)...  If you considered a point on either side of that theta value, say theta + episilon and theta - episilon.  If you figure out the slope of the line between those two points (rise over run between the two points: y over x), you can compare that to the derivative.  To abstract that out to a bunch of thetas, you can approximate the partial derivatives with respect to a single theta like this:

$$ J(\Theta) = \frac{J(\theta_1 + \epsilon, \theta_2, \theta_3....\theta_n)}{2\epsilon} $$
$$ J(\Theta) = \frac{J(\theta_1, \theta_2 + \epsilon , \theta_3....\theta_n)}{2\epsilon} $$

etc, up to:

$$ J(\Theta) = \frac{J(\theta_1, \theta_2, \theta_3....\theta_n + \epsilon)}{2\epsilon} $$

I think that means that you're calculating the gradients for each partial derivative through back propogation and then again via this formula, as many times as you have thetas in an given activation node, in any layer.  But how did this go with regular regression and simultaneous update?  You calculated the partial derivates with respect to each theta, one at a time and then updated all thetas at once (simultaneous updates). And the result was a new set of thetas.  But the derivates you calculated were partial - one for each theta in the hypothesis while the rest remained unchanged. So, what we're doing here is the same, and that gradient checking is once for each theta value for each activation node in each layer (except the last (predictions) and the first (x's)).


## Regularization of gradients in neural networks

You can add regularization after the gradients are calculated via back propogation in a neural network.  Just multiply each theta (except for the 1st (bias) column of thetas) to the corresponding gradient found through back prop.  This should be done for each layer's Thetas.

$$ \Delta^{(l)} = \Delta^{(l)} + \frac{\lambda}{m} * \Theta^{(l)} $$

for all non-bias $\theta$'s. So, you add zero to the bias gradients.


## Validating your model

So, you've trained your model.  And you can't get it to perform very well on the test set.  Some things you could do:
    1. get more training data
    2. add features (either more columns from real data elements OR exponential elements from basic data values: $x_1^2, x_1^3, x_2^2, x_2^3$ 
        ....etc.)
    3, remove features
    4. adjust lambda (the regularization parameter)

And now it's a bit better!  But maybe it's now overfitting to get that better performance.  How do you know?  Split the training data into a training, validation and test set.  Without a validation set, a good split is generally 70/30 (training/test).  With a validation set, the general practice is to split 60/20/20 (train, validation, test).  

Then train using different versions of the model against the training set and compare the performance of the models by running them with the learned parameters against the *validation* set.  You can compare the performance by looking at the cost arrived at by each AND getting an error ratio - the number of incorrect predictions divided by the overall number of samples in the validation set.

Sometimes models can have high bias or high variance or both.  Bias is underfitting, variance is overfitting.

In order to tell what's happening and how you might want to adjust your model, look at the differences between costs for the training set run and the validation set run.  If the cost is high for both runs and about the same, it's most likely you have a case of high bias (underfitting on the test set is likely to lead to underfitting on the validation set).  If the cost is considerably lower for the training set than for the validation set, it's likely that you have a case of high variance (overfitting matched the training set too closely and did not generalize well to the validation set).

Likewise, a large lambda value (regularization parameter) can also cause high bias (underfitting) and a small value.  A too large lambda will penalize all thetas and tend to flatten the hypothesis (think of a simple linear regression hypothesis that you can graph).  So how to choose a good lambda? You can iteratively just try new lambda values.  Ng recommends using 0, 0.01, 0.02, 0.04, 0.08 and training again, saving the resulting $\Theta$ vectors for each lambda.  Then try those thetas on the validation set and see which gives the lowest cost.  You might even want to graph the cost against lambda as lambda increases and then do the same for the validation set with the thetas that resulted from each training run.  Then you'll be able to tell where lambda might be best for the validation set.

You can further find out what's ailing your model (overfitting or underfitting, high variance or high bias) by graphing "learning curves".  You can graph the error percentage for both training and validation data as the number of samples increases.  If, as the number of samples increases, the test and train curves are both relatively high (compared to what?), and pretty similar, it means that you have a high bias problem (underfitting).  That's evident because a) the error is high and b) high bias problems in the model don't get better on a different set of data.

If, on the other hand, as the number of samples increases, the training data error is pretty small and the validation error remains in a higher and unsatisfactory (compared to what?) range, you have a case of high variance.  A model with high variance will match the training set well but not generalize well to unseen data.  But, with this curve, the two error rates will converge if you add more training data to train the model.  Why is that?  I think it's because the training data will then be more similar to the validation/test data.

Just keep in mind that you train with the training set and then use those $\Theta$'s to measure the error across all validation set examples, totaled and divided by the total number of validation set examples to arrive at an error rate.  You don't train against the validation set.

