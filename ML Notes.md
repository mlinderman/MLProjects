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

Also very important: the cost function to do iterative minimization has a regularization component (for n > 1).  But if you use the same cost function to measure cost of the learned thetas against another data set (validation or test), you should set lambda to zero. Remember, regularization artificially inflates the value of thetas to prevent overfitting.  That lambda has no use, therefore, when you're just measuring how well the calculated thetas work against validation or test data.  So set the lambda to zero when calculating the error (cost) of thetas against test or validation sets.

## ML Systems design

Consider Ng's example of a spam classifier.  A pointer about that:
1. The most common words from the data set, maybe 1000 or 10000 of them, represent features.  (I'd assume you'd exclude articles and conjunctions, etc. from that list.)  And each email (a sample), would have 1s or 0s for those feature values.

- How to spend your time?
    - collect lots of data. (doesn't always help, as we've seen)
    - develop sophisticated features (email routing from the header, for example) by understanding how spammers route emails.
    - email message body examination via algorithms - refine the features or create new ones - like concerning the use of exclamation points or deliberate mispellings

bottom line, it's hard to tell which might help.  But you shouldn't just spend time on something you have a gut intuition about

Instead, use error analysis by building a simple model first - quick and dirty.  With that, you can plot learning curves to detect high bias or high variance to help decide how to proceed.  This avoids premature optimization.  Also, you can get a single metric to test things against - say it's the error rate.  Then you can tweak to see what might affect that error rate.

Error analysis: look specifically on what the basic algorithm is misclassifying.  That can guide you.  Given the errors made (using the spam filter example), try to categorize these (pharma, fake watches, phishing emails) then you can build new features to help categorize the largest error sets.
It's important to have a numerical way to measure performance and use it for each test, rather than eyeballing the errors.  Do error analysis on validation set rather than test set.

### Handling skewed data

Skewed classes are when a classifier is trying to detect a very small occurrence.  Say, for example, that only .5% of examples are positive. You could write what appears to be a great algorithm that gets 99% of the correct predictions but that still misses half of the positives.  So it's not as good as it seems.  A non-learning algorithm that just predicts every sample is negative would do better since that would have a .5% error.

This is the problem with using accuracy as a measure. And also a problem with using a single measure to gauge algorithm performance.
Enter precision vs. recall

Precision:  true positives / predicted positives

OR:         true positives / true positives + false positives

Recall:    true positives / true positives + false negatives

high precision and high recall are good.  But they move in opposite directions.  So, you want to measure both.  (y = 1 should be the rarer class).  Use both precision and recall, especially when classes are very skewed.
Depending on what sort of behavior you want - whether you want to get as many cases correct or not miss any - may mean that you want to change the logistic regression threshold from 0.5 to a higher or lower number.  Making it higher, say .9 for positives means you'd make fewer false positive guesses but possibly miss some cases (higher precision).  Making the threshold lower would mean you would miss fewer cases but also that you'd falsely categorize some as positive (higher recall).  Graphing precision vs recall can result in different looking graphs but they do have an inverse-ish relationship.

So, how to use these to find the best algorithm.  We now have 2 real measures of the algorithm's performance.  If you have to wonder about which is better, it can be difficult.  So, to get to a single metric again from these values, you could take the average but that can give you a higher average even if precision and recall are very different.  Instead, use the F or F1 score:

$$ 2\frac{PR}{P+R} $$

There are other measures but this is the standard in ML circles. F scores are between 0 and 1.  So, manipulating the threshold for logistic regression and seeing which has the highest F score against the validation set would be a pretty good way to pick the best threshold.  Ng didn't mention just using the f-score when you're not manipulating the logistic threshold but I've read about f-scores before in articles that don't mention the logistic threshold.  Bottom line, it's just a way to make sure your awesome ML model isn't actually just predicting the same thing against a data set with skewed classes.

### Handling Large Data Sets

Under certain conditions, training against a lot of data is helpful (not always, like when your algorithm has high bias - underfitting) when:
    a) your y's can be predicted pretty accurately by a human expert given the features (like the correct form of a word (y) to use between two halves of a sentence (x's) (too, two to)).  
    b) you have a lot of features (suggesting you'll be able to achieve low bias - since more features tend to overfit, not underfit).  In this situation, a large training set is likely to get you low variance.
These are actually similar since more features provides more information to an expert human.

Precision = true positives / (true positives + false positives)
Recall = true positives / (true positives + false negatives)

P = 85 / (85 + 890) = .08717949
R = 85 / (85 + 15) = .85
F = 2 * (P*R)/(P+R)
  = 2 * .14820513 / .9371949


## Support Vector Machine (SVM)

This is a powerful supervised learning algorithm that's powerful and popular.  Looks like logistic regression with some small changes to the cost function:

$$ -(ylog \frac{1}{1 + e^{-\Theta^Tx}}) - (1 - y)log(1 - \frac{1}{1 + e^{-\Theta^Tx}}) $$


If you remember what this is doing, it's basically two different functions, 1 for when y = 1 and another for y = 0.  The result is two different cost functions, one which nears a limit of 0 as $\Theta^Tx$ gets larger than 0 and another that approaches a limit of 0 as $\Theta^Tx$ gets smaller.  Those functions are curved.  Support vector machines use a slightly different set of cost functions that are a combination of straight line segments that roughly follow the same contours as the logistic cost functions.  That's computationally cheaper, apparently.

Support vector machines are also known as "large margin classifiers."  That's because SVMs maximize the distance between positive and negative training examples and the decision boundary.  There's a long explanation about that on an "optional" lecture by Ng.  It involves vector math and "norms" (euclidian lengths of vectors).  It's interesting but don't think it's something you're going to be building.  For that matter, it sounds like Ng recommends just using built-in SVMs.  

Conventions for expressing the cost function for SVMs are a little bit different in addition to the elimination of the log functions above.  The $\frac{1}{m}$ is removed from both the first and 2nd (regularization) terms since it's considered a constant.  

Also, lambda $\lambda$, is moved from the regularization part of the equation to the cost part.  And it's called C.  And because it is moved, it behaves in an opposite manner than $\lambda$.  That is, a large value for C would equate with a tendency to underfit, rather than overfit.  And a small value would tend to make the hypothesis overfit.  So C is like $\frac{1}{\lambda}$.  And as long as C isn't too large, SVMs do the right thing when there are a few outliers in the sample set where positives or negatives are mixed in with the others.

Here's what all those changes look like together:

$$ min_\Theta\;\;  C\sum_{i=1}^m[y^{(i)} cost_1(\Theta^Tx^{(i)}) + (1 - y^{(i)}) cost_0(\Theta^Tx^{(i)})] + \frac{1}{2} \sum_{j=1}^n \Theta_j^2 $$


Those $cost_0$ and $cost_1$ functions are representations of the function that's made of 2 line segments as mentioned above.  Not sure why Ng just doesn't write them (can they be that hard?) unless again, it's because he doesn't want to get into it because he's going to recommend pre-packaged SVMs.  

Also in the equation above, notice the removal of the $\frac{1}{m}$'s and the C variable.  My guess is that we'll need to supply the value of C to pre-packaged SVMs.

By the way, due to the replaced cost functions (I think) and SVM just predicts 0 or 1 depending on the value of $\Theta^TX$.  Due to the SVM function, $\Theta^TX$ has to be > 1 or less than -1 for it to predict 1 or 0.

### Kernels
This topic was introduced as part of the SVM lectures so it must be that SVMs use kernels?  In any case, Ng introduced the idea of a Gaussian kernel (one example of a kernel) that he used to create new features.  Think of a 3D Gaussian curve where the highest point of the curve represents proximity to something else.  And the values you're measuring nearness to are the other samples.  In order to avoid creating a bunch of polynomial features to fit a model, you could use kernels instead.  Ng recommends creating new features for each sample that are Gaussian kernels representing similarity to other samples.  And then you replace the original features with the new kernal features, 1 per sample.

The Gaussian kernel looks like this:

$$ exp(-\frac{||x - l^{(i)}||^2}{2\sigma^2}) $$

where $\sigma$ is a variable.  That numerator is the Euclidian distance between x and the "landmark" chosen at random (another sample, in this case). The numerator can also be written as a $ \sum $

The closer the landmark is to x, the closer the feature is to 1.  The further it is, closer to 0.  The $\sigma$ changes the rate at which the Gaussian function goes to zero as you move away from the landmark.  

So, Ng says for landmarks you use all the samples, 1 landmark per training example.  Oh, so you're really measuring how close a *new* example (unseen data), is to training examples that you know the y's for.  So, you make a vector f of new features and treat it exactly like an X, including adding a $ f_0 = 1 $. And then you use $\Theta^Tf$.  Finding parameters is just like before but with $\Theta^Tf$:



$$ min_\Theta\;\;  C\sum_{i=1}^m[y^{(i)} cost_1(\Theta^Tf^{(i)}) + (1 - y^{(i)}) cost_0(\Theta^Tf^{(i)})] + \frac{1}{2} \sum_{j=1}^m \Theta_j^2 $$

(Notice the f replacing x in both cost functions and also the m replacing n in the regularization part.  Since we've created a feature for each sample, now n = m or the number of samples is the same as the number of features except for the interceptor $f_0$)  Also, as an aside, most support vector machine implementation change the regularization part to be $\Theta^T\Theta$.  That and some other trickery makes it possible for SVMs to be more efficient at computation over problems with a large number of features (10, 10k).  SVMs and Kernels work well together but kernels don't work that well with logistic regression - it'll run slowly.  Bottom line, use software packages instead or rolling your own.

### Bias and variance trade-offs when using SVMS
Large value of C or small value of $\sigma^2$ = lower bias, high variance, overfitting tendency
Small value of C or large value of $\sigma^2$ = higher bias, lower variance, underfitting tendency

### SVMs in practice
Need to choose C, sigma and kernel to use
  - linear kernel is no kernel at all - use if you have a large number of features, small number of training set where you might risk overfitting
  - Gaussian kernel - if you choose this, you need to also choose a $\sigma^2$ - use for complex, non-linear hypothesis.  You'll have to provide a function to compute the kernel.  Wow, these will automatically compute the features from this function.

Choosing the best C and sigma values can be done (as in the homework) by training against a range of values for each - all combinations.  During that iterative training, you keep track of the C, sigma and percentage of missed estimates each combination's model produces against the *validation set*.  After all combinations' results are recorded, find the minimum missed estimate percentage in (presumably) a vector or matrix you used for that purpose and use that value of C and sigma to train the best model.  If this iterative training took a long time, you'd obviously want to save the models that resulted for each so that you wouldn't have to do it again for the final version.  

It's important to do feature scaling because a Gaussian kernel computing differences between x and l (landmarks), could be very different. (Consider house square feet vs. number of bedrooms.)  Not all similarity functions make valid kernels.  A few valid others are: Polynomial kernel: k(x, l) = $(X^Tl)^2$.  That's just one version - but usually you need to provide the degree of polynomial and a constant that's added to that equation.

But all valid kernels are so-called similarity kernels.  String kernel, chi-square, etc.

How to decide whether to use SVM or logistic regression?  Ng equated logistic regression and an SVM with a "linear" kernel (no kernel).  The only case where he recommends an SVM with a Gaussian function is when n is small (< 1000 or so) and m is of intermediate size (< 10,000).  What about neural networks?  They can work well under most conditions but could be slower to train.  For SVMs, you don't have to worry about local minima since the



# Unsupervised Learning
Unsupervised learning problems have no labels, no y's.  So you try to find some structure in the dataset.

### K-Means clustering
This is one way to find structure/patterns in the data.  K-means is the most widely used clustering algorithm.  Clustering helps you group the data - like market segmentation, social network analysis, organizing data centers better (see which servers talk to each other the most)

For k-means clustering, you pick K random centroids (these are usually randomly selected samples from the dataset).  And then you assign samples to clusters based on which centroid is nearest.  After that, you re-locate the centroids to be at the average of its assigned samples.  You do this iteratively for some number of repetitions or until the cost stabilizes (the centroids don't move that much).

1.  Randomly initialize K centroids
2.  Assign samples to centroids
3.  Move centroids to the average of all the assigned samples
4.  Repeat steps 2, 3 for n iterations

K-means algorithm, more formally:
 - don't use a $x^{(0)}
 - $\mu$ is used to denote cluster centroids, $\mu_1, \mu_2, .....  \mu_K$
 - to measure distance to each centroid, you should be able to do vector subtraction and element-by squaring and then summing.  Here's the distance:

 $$ ||x^{(i)} - \mu_k||^2 $$

 The cost function is then mean squared differences of distances between samples and assigned cluster centroids:

 $$ \frac{1}{m} \sum_{i=1}^m||x^{(i)} - \mu_{c^{(i)}}||^2 $$

To move the centroids, find the mean of all the assigned samples.  This must mean that you're finding means for each term of the samples assigned to the cluster.  If any clusters get no samples assigned, you should just eliminate that cluster OR randomly re-assign centroids and try again.

The result of these iterative moves of cluster centrods then is to attempt to minimize the cost function - also called the "distortion" of the k-means algorithm.

When you do the random initialization of K's, you may unluckily choose K's that don't result in minimization to a global minimum but to a local one.  In order to avoid this, you should pick randomly multiple times measuring the cost for each initialized set of Ks.  And then choose the initialization set of K's that have the lowest cost before starting the iterative process that adjusts the location of the Ks.  You would typically do this between 50 and 1000 times.  With a number of clusters between 2 and 10, this works pretty well.  With higher number of Ks, it won't make much of a difference, however, but I think you're also less likely to find local optima with lots of Ks.

Choosing number of clusters (K): you can use the elbow method (finding where the cost of clustering levels out when graphed against the number of clusters). That's not always a clear point, however.  Or you can ask about the purpose and what number of clusters would best serve the purpose.  In the t-shirt size example - finding size ranges to make up XS/S/M/L/XL, maybe it makes more sense to support fewer sizes (S/M/L) because it might be cheaper to manufacture.  In the end, often by-hand choices for K are often the most practical.

### Dimensionality Reduction
This is another unsupervised learning algorithm.  It serves the purpose of finding highly correlated (linearly related) features and making a single feature out of those by combining them into a different feature.  So one number takes the place of two in Ng's case of rounded cm and rounded inch features.  The end result is that your final learning algorithm will take less memory and run faster. 

With 3 dimensions, you can possibly reduce them to 2 if you can determine that they are correlated roughly to the same plane (a 2d space).  With both the 2d and 3d reductions, you need to "project" the data onto fewer dimensions. And it's important to note that finding a line through 2 dimensional data like this with dimensionality reduction is *not* the same as finding a line through the data as with linear regression.  Why not?  Well, Ng described the difference like this: projection finds the perpendicular distance (shortest) to the line whereas linear regression finds the vertical distance to the proposed line.  Ok, I'm willing to accept that because in linear regression, it's the difference between what the hypothesis predicts and a label (a y value) for a given X sample.  But with dimensionality reduction, there are no y values (it's unsupervised learning) so you're not interested in the difference in the y predicted and the y supplied but instead just the shortest distance to the line - the projection.  And no particular dimension is treated differently from the rest.  I think what that means is that there are no weights in PCA but see below about how PCA is performed.

With dimensionality reduction, you can even take 50 dimensions down to 2 or 3 with the resulting dimensions not really having any meaning but making it possible to visualize the data by plotting it and gaining some insights about it in order to eliminate or consolidate dimensions.

#### Prinipal Component Analysis
Find a lower dimensional surface onto which to project the data (a line or plane) so that the sum of squares of distances to the surface is minimized.  By the way, feature scaling is important to do before you project data onto a surface.  So, objective is to find a vector onto which to project the data so that it results in small "reconstruction" errors.  To project onto 2 dimensions or 3, you'd need to find 2 or 3 vectors, respectively, describing the surface.  The objective is to miminize the projection error.

##### The PCA Algorithm
You must do mean normalization and feature scaling first.  That's the same as before but seems unfamiliar now.  Mean normalization will just take the average of the feature values and find the mean and then replace the feature value with the feature value minus the mean:

$$ \mu_j = \frac{1}{m} \sum_{i=1}^m x_j^{(i)} $$

to find the mean, then:

$$ x_j^{(i)} = x_j - \mu_j $$

to replace each sample's value for that feature with a mean-normalized value.  Ng says that then all features will have a zero mean.  Hadn't thought of it that way but I guess it's true since all values will have the mean subtracted, if you calculated the mean again, it would be zero.

Then do feature scaling, same as before with supervised learning:

$$  x^{(i)}_j = \frac{x^{(i)}_j - \mu_j}{\sigma_j} $$

Sigma is the standard deviation of the feature.  But other values (like max minus min) can be used less commonly.  Again, this is the same as before.

Keep in mind that the goal is to translate a point in 2 dimensional space (x,y) to a single value in 1-dimensional space (a line).  For 3 to 2 dimensions, we'll get (x,y) from (x,y,z). So, you need to compute the vector or vectors onto which to project the data.  Turns out, the math is very complicated but the procedure isn't.  The mathematical proof Ng did not go into.

So the procedure is that first, you compute the "covariance" matrix (called Sigma, which, confusingly, is not a sum operation though it looks a lot like one and is also *not* the standard deviation). And then use that matrix called Sigma to compute the "eigenvectors" of that matrix.  Say what?  He glossed over the details but computing eigenvectors is the same as a "single value decomposition".  You can do this in octave using the svd function.  There are equivalents in other languages (like Python, I'd assume).  The octave svd function, returns 3 vectors, U, S and V - or maybe that's a matrix...

So, to get into the details, to compute the covariance matrix, you take each sample *column*, or feature (not rows this time), and multiply it by it's transposed version (producing an n x n matrix) and then add all the matrices that result (1 per feature)together.  The result is still an n x n matrix. (Don't forget: n is the feature index and m the samples index.)  Averaging that over the number of samples ($\frac{1}{m}$) just gives you a different n x n matrix.  That's $\Sigma$.    

$$ \Sigma = \frac{1}{m}\sum_{i-1}^m(x^{(i)})(x^{(i)})^T $$

That's from Ng.  But it can't be right since he claims it'll result in an n x n matrix.   But if $x^{(i)}$ is a single example, a row, as it's always been throughout the course, then this will result in 1 x 1 vector.  (I think he must mean to reverse those two terms, transposed one first.) It doesn't matter since the vectorized version below does look like it will return an n x n matrix.  So, moving on, the next step (in octave):

$$ [U,S,V] = svd(\Sigma) $$

And what you get as a result is 3 things and you care only about the U matrix which, it turns out, is also an n x n matrix.  And each column of that U matrix is the vector we wanted to find to project the features onto to reduce them.  Then, if the goal is to project n dimensions down to k dimensions, you then just take the first k columns of the U matrix so you have an n x k matrix that we'll call $U_{reduce}$  To arrive at our new reduced feature set z, you then multiply transposed $U_{reduce}$ by X where (this is critical), X is a single column (one feature) of X as in the Sigma calculation above.  That's an n x k matrix times an n x 1 matrix.  The fact that you're transposing $U_reduce$, however means that the matrix will be k x n when it's multiplied.  And k x n multiplied by n x 1 would equal k x 1 vector and that's your new feature - same from 1 to k features.

To summarize the math for the complete PCA, after mean normalization and feature scaling:

Vectorized sigma calculation (from above):

$$ \Sigma = \frac{1}{m} * X^T * X $$

Note that that will result in an n x m times an m x n which will result in an n x n matrix, just like the non-vectorized equation above.  So that much matches, at least.  

Then:

$$ [U,S,V] = svd(\Sigma) $$

U from that vector is a n x n matrix and you take the first k columns to make a new matrix which we'll call $U_{reduce}$ and is n x k.  Then, understanding that Z is the reduced feature set we're looking for:

$$ Z = X * U_{reduce} $$

Remember that $U_{reduce}$ is n x k.  And X is m x n.  So the n's line up and you get a m x k matrix which is exactly what we're looking for - a reduced feature matrix.

##### Recreation of the original matrix (approximately)
Turns out, PCA can be used to re-create an approximation of the original (say 1000 dimensions) from 100 dimensions produced by PCA.  Like this:

$$ X = Z * U_{reduce}^T $$

That's m x k times k x n which will yield an m x n matrix, the dimensions of the original.  By the way, I had to manipulate this a bit to get that result from what Ng proposed.

##### How to pick the number of dimensions
How to pick k (the number of dimensions to reduce to) anyway?  You pick it so that 99% of the variance is maintained.  What? PCA attempts to minimize the average squared projection error (the distance between the original feature and the surface to which it's projected):

$$ \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - x_{approx}^{(i)}||^2 $$

Another definition that's useful here: total variation in the data (how far the training examples are from being all zeros) - having nothing to do wth PCA but useful in choosing k.  The formula for that is:

$$ \frac{1}{m} \sum_{i=1}^m ||x^{(i)}||^2 $$

And you use these two formulas together to calculate the "% variance retained" after PCA is performed and ensure that it's 99% or greater:

$$ \frac{  \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - x_{approx}^{(i)}||^2  }{ \frac{1}{m} \sum_{i=1}^m ||x^{(i)}||^2 } <= 0.01 $$


So, PCA can be used to speed up a learning algorithm - very common to do this, apparently.  If you had 10k features, for example, from 100x100 images.  That could make for a slow algorithm, no matter what technique you use.





