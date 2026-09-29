# CONST things to experiment with
 - hyperparameter tuning



# DNN

## i'm encountering an issue with early stop being called to early
i can try removing early stop
see what that does
i'll try it
our goal right now is to make it look like it's good

there's certainly something that we're missing
we're worried about prediction, so we can look at the model structure to make sure it looks good
we have an input layer, 8, then output
i think
i assume the input layer is added automatically
or something like that
but we know we have to use a feed forward neural network
https://machinelearningmastery.com/using-normalization-layers-to-improve-deep-learning-models/
hmm normalization
what if i just add one to the model

okay adding the normalization layer helped a lot
i can get a r2 of .5, but i'd like something over .8
let's do some more research, but maybe we should use ai
we also have to consider that i'm overfitting linear regression, and perhaps to explore that more

https://datacalculus.com/en/knowledge-hub/data-analytics/data-cleaning-and-preprocessing/data-preprocessing-for-neural-networks/
maybe i can also clean the data more
maybe the setting n/a to zero might not have been the best idea
determining outliers too


# preprocessing
first we need to read a csv file
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html 
let's practice in the uhm. terminal
train.dtypes - see datatypes
df.dropna() to drop na - maybe. we'll try it without and compare 
print(df[df["Code"] == "IN"]) - filtering rows
df[['First', 'Last']] = df.Name.str.split(char, expand=True) - delimit
data['result'] = data['result'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC')) - trim characters
df.shape - get dims
df.drop(columns=["B", "C"]) - drop columns
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html - for replace
not really sure how it works
how to tell if a column has any NaN
change the dtype of the new columns
df.replace(old, new) to replace 
df.isnull(), df[col].isnull() - returns df with False/True and can aggregate from there
e.g., df[df[col].isnull() == True]
df.isnull().any() to see which columns are teh culprits
PctPrivateCoverageAlone - 480
PctEmployed16_Over - 123
PctSomeCol18_24 - 1828

# multiple linear regression
## failed research
https://statisticsfundamentals.com/multiple-linear-regression/
  r^2 = 1 - SSres/SStot (residual sum of squares, and total sum of squares)
  adjusted r^2 = 1 - (1-R²)(n-1)/(n-p-1) (penalized for number of parameters)
https://365datascience.com/tutorials/statistics-tutorials/sum-squares/ 
  sum of squares total (sstot) = ssres + ssreg
  sum of squares residual (ssres) = sum((y-yhat)^2)
  ssreg = regression sum of squares = sum((x-xbar)^2)
  wait that doesn't work

## how to handle b with different feature sets? what is 
oh it's supposed to be a single value

## WHY IS MY ERROR GETTING BIGGER????
well first we should take yhat and input it to optimizeSGD because less computation
https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/ 
maybe my y and yhat are reversed
when are her office hours
they're by appointment at Digital Futures

https://aimltutorial.in/lesson/gradient-descent-mlrmultiple-linear-regression/
important

i got it to work a lot better, if i turn the learning rate way down to like 0.0001
https://developers.google.com/machine-learning/crash-course/linear-regression/hyperparameters
okay so if the learning rate is too big, it bounces around wildly
my learning rate was 0.01, that was too big ig
what if the learning rate changes with the weights?
e.g. hyperparameter tuning
so if dldw is 1000, maybe i want w to change by 1
if it's 10, maybe 0.01
so dldw/1000?
let's try it
it gets really slow near the end. like a lot slower
having the constant was a nicer balance, perhaps
i wonder if we can measure that

## problem - the association is really bad
i mean, maybe it's not a linear relationship at all?
well isn't DNN just glorified linear regression?
kinda sorta. probably not really 
i need a plan
well experimenting with hyperparameters can maybe be used to try to improve things
i can probably speed it up a bunch by removing console output
i can try R2-adjusted - but that might just make it smaller
i kind of want to investigate what makes SSTO different from the two calculations
or if i'm calculated R2 right
we probably are, we just might need to use DNNs
we can try with that, then we can play around with different ways of optimizing it. perhaps
so move on to DNNs? yeah, and if they're a bust we can always go back to optimizing LR
but it's likely that the data isn't linear. 
oh we might have to try a different loss function as some point. 
we have time
https://www.geeksforgeeks.org/machine-learning/epoch-in-machine-learning/ 
perhaps it's overfitted?


# techniques in class
(these might be useful to consider if we build the models and it turns out that they are really bad)
 - model validation
   - leave p-out cross validation
   - k-fold cross validation
   - hold-out
   - repeat of subsampling 
 - activation functions (for DNN)
   - ReLU
   - Leaky ReLU
   - sigmoid
   - tanh
   - softplus
   - ELU
   - SELU
   - switch
   - GELU
 - regularization
   - weight regularization
     - weight decay
     - L2 regularizer
     - L1 regularizer
     - Elastic Net
     - max-norm
     - orthogonal regularization
   - data-based regularization
     - augmentation
     - random erasing
     - text augmentation
     - feature space augmentation
   - noise based regularization
   - label smoothing
   - drop out based
     - dropout
     - drop connect
     - spatial drop out
 - early stop
 - learning rate scheduler
 - momentum

# what is a validation set and what is it used for 
https://www.geeksforgeeks.org/machine-learning/training-vs-testing-vs-validation-sets/
it looks to be about the same size as the test set?
it fine-tunes hyperparameters and prevents overfitting
i'm still confused how to implement it?
the code example isn't very good. it just outputs the accuracy of the x against the y
so it's used during training, but when? does it have a higher learning parameter?
do the different hyperparameters react dynamically depending on the test against the validation set?

https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets
it can be used at the end of training for early stopping, when the error begins to increase
a list of things in here, about common errors
so if i'm struggling to get the proper accuracy, i should probably peer into data validation and all that

https://www.techtarget.com/whatis/definition/validation-set 
maybe as i go about the model i can list the hyperparameters then scheme about how i want to use the validation set to tune them
understanding the hyperparameters will be the precursor to that

https://medium.com/@meisshaily/the-ultimate-guide-to-validation-sets-in-machine-learning-107d26cb08a1
locked

https://www.articsledge.com/post/validation-set 
"Validation enables hyperparameter tuning. Learning rate, tree depth, regularization strength, dropout rate, number of layers—none of these are learned by the model. They are chosen by you. The validation set is the only fair way to compare different hyperparameter configurations."
used for early stopping
can be used to choose between models
"Adjust: change hyperparameters, modify architecture, add regularization, engineer new features." - done after running on validation set and analyzing
then train again until reached an ideal accuracy
overfitting is when training accuracy and loss are really high/low, but in validation it's low/high

https://www.articsledge.com/post/hyperparameter-tuning
hyperparameter tuning
not strictly necessary on most runs, but for some models hyperparameter tuning gives up to 15% accuracy gains
it seems that hyperparameter tuning doesn't seem strictly necessary, but we can experiment with it
but it seems that the validations set will mostly be used for early stopping, when the error starts to increase perhaps

# HW1
 - submit ai prompts
 - canvas submissions
   - zip containing two reports
     - homework report containing: 1) answers to 12 questions, 2) table in step 2, and 3) performance plot in step 5
     - report containing all the prmopts
   - all code 
   - model weights of the best model that you used to submit prediction on kaggle
 - train.csv needs to be split into training and validation
 - submission.csv test file, id and TARGET_deathRate columns
 - R^2 is the evaluation technique, probably derived from the submissions.sv file
 - models tested - linear regression, DNN-8-output, DNN-16-8-o, F-16-8-4-O, F-30-16-8-4-O. five in total. 
 - data, models questions, 
 - use MSE as the loss function, but also will need to try a different loss function
    - so each of the models need to be able to swap out the loss function
 - will only use SGD, but not a bad idea to make optimization swappable too


 okay. i will probably need to use pandas for data imports and matplotlib. probably numpy too. pretty much can turn anything into a numpy array

 i'll start with directly working on the models

 first i need to know what a validation set is and what it does

 then we can work on designing linear regression and figuring out what knowledge i'm missing there. shouldn't be much, as i have notes and i've done it before. but that would probably be next. 
 and getting that all set up