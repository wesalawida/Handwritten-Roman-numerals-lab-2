# Handwritten-Roman-numerals-lab-2
# LAB 2

##### Manar Awida

##### Gadiel Eitan

##### github : https://github.com/wesalawida/Handwritten-Roman-numerals-lab-

# Our first experience :

##### Our 'go-to' was to filter mislabled training images based by training some classifiers and use their predictions (noise elimination). In other words. Mark every instance in the training set as mislabeled (1) or not (0). Then, we tried 3 methods:

#### 1. Filter by majority vote label (the one which most of the claafiers predicted). If this label differs from the 'real' given label we omit this sample.

##### 2. Filter by consensus label. Instance is marked as mislabeled if all of the learnes (algorithms) tagged it as mislabeled. We tried 6 algorithms and changed them a bit during our analysis. There is an easy tutorial here: https://longjp.github.io/statcomp/projects/mislabeled.pdf Note: we didn't use the third method (where filtering is done by one algorithm) because it's doesn't seem accurate and we need some prior bound on the % of "label-errors" and more data to train and being able to generalize, let alone this noisy and complex data. This topic also delas with some formulated errors which we couldn't calculate in our case (depends on prior info on the noise or similar assumptions).

#### Unfotunately, all the methods failed because the predictions were bad (almost 1950 were wrong). The main reason is our shallow learning on the complicated data (with small sample comlexity) which led to underfitting. We tried to train a simple fully connected netowrk which led to overfitting as expected. Again, the given dataset is too small for this. It works

# All the process in thr PDF file 

## The Rustles:


#### 1 - The training loss and the validation loss seem to reduce and stay at a constant value. This means that the model is well trained and is equally good on the training data as well as the hidden data.

#### 2 - From the plot of accuracy we can see that the model be trained good as the trend for accuracy on both datasets is rising and stable with the epochs increasing.
