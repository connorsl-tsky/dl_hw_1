# preprocessing

# linear regression
Explain
    We have a set of features x, and labels y
    We assign random numbers or initialize the weights and bias to some initial values HYPERPARAMETER?
    we predict y (y-hat) with y-hat=W*(matrix mult)X + B
    We compute the loss with the loss function, e.g. MSE 
    L = 1/n sum(i=1, n)([Y-hati - Yi]^2)
    then we apply stochastic gradient descent via Th(theta) <- Th - I(ita, learning rate)Grad(L)
    where W and B are Th's
    Ita is between 0 and 1
    how do we compute the gradient? dL/dW? 
    https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/ 
    can add lasso, rdige, elastic net regularization
    https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/
    thank you gfg i love you
    dL/dw = 2/n sum(i=1, n) x((wx + b) - y)
    dL/db = 2/n sum(i=1, n) (wx + b) - y
    then we update with 
    w <- w - I * dL/dw
    b <- b - I * dL/db
    via stochastic gradient descent
    that trains it
    now we can verify after training
    by running the prediction with the weights on the training data
    then we can determine the....error?
    yeah the loss
    and we compare the with the last error, and if it goes up it we stop
    or something
    then we can run on the testing data 
    so we train it on the testing data, compute the y-hat and do the learning rate
    then we verify it on the validation set?
    we can try it on every validation and see how fast it is

LinearRegression(learning rate, early stop)
    init()

vars
    learning rate
    early stop

    weights 2d array
    bias 1d array

initialize()
    init weights to random numbers
    bias to random numbers

precondition 
    call init() 
train(x-train, y-train, x-val, y-val) -> void (updates w/b)
    prev error = curr error ? curr error : 0
    curr error = validate(x-val, y-val)
    while (prev error - curr error) > 0
        yhat = predict(x-train) 
        w, b = optimizerSGD(yhat, y-train)

predict(x) -> y^
    y-hat=W*(matrix mult)X + B

lossMSE(y^, y) -> int (error)
    L = 1/n sum(i=1, n)([Y-hati - Yi]^2)

optimizerSGD(y^, y) -> w, b
    dL/dw = 2/n sum(i=1, n) x((wx + b) - y)
    dL/db = 2/n sum(i=1, n) (wx + b) - y
    w = w - I * dL/dw
    b = b - I * dL/db

validate(x-validation, y-validation) -> int (error)
    yhat = predict()
    return lossMSE(yhat, y-val)

test(x-test)   
    yhat = predict(x)
    computeR2(x, yhat)

computeR2(x, y) -> float 
    n = len(x) = len(y)
    compute sumx, sumy, sum(x^2), sum(y^2), sum(xy)
    return n*Sum(xy)-(SumX)(SumY) / sqrt([nSum(X^2)-(Sumx)^2][nSum(y^2)-(SumY)^2])



# DNN