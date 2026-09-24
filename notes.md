# CONST things to experiment with
 - hyperparameter tuning

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