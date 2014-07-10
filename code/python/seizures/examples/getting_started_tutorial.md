# GATSBY HACKATHON GETTING STARTED TUTORIAL


So far, the approach we took is the standard (in the ML world at least) two steps classification task where **features** are extracted from data _X_ and passed through a **decision rule** or to get a binary output _y_. Various proxies (one per method) are used to get a **predictor** returning p(_y_=1|_X_).

This can be done with various features and various decision rule.  
It is quite easy to add extra decision rule or most importantly new _features_ on which to achieve the classification.

In this doc are described

* the task a bit more precisely
* what is in the framework so far and discuss ideas of extensions
* where is the data, how is it structured and how to play with it.
* how you can readily implement features and test them yourself 


## Task

For an input corresponding to 

* a patient ID
* a (#channels)*(#time points) matrix corresponding to 1s window of simultaneoulsy recorded EEG signals accross channels.

Predict

1. whether the window correspond to a seizure events
2. if seizure, whether it is early or late in the seizure event (>15s after seizure onset)


## Framework

### What is in the framework?
For a _predictor_ and _feature_

* training of the decision rule from the training data (one rule per subject)
* Cross-validation (LOO: Leave One Out) to test performance (AUC: Area Under the Curve criteria) of _features_ on training data.

### What can be added to the framework?

* hyperparameter optimization: ex in svm: what is the penalty cost value, what is the norm used for the penalty
* merging of features: if you give features F1, F2: test if any combination (stacking, sum ...) of those leads to any improvement in classification performance.

### How can you readily contribute to the science?

* build your own _feature_, test in on training data through cross validation


## How do I do in practice?


### Setting up the tools

I here describe in details how to get started on a linux machine (for ex your gatsby desktop)

You need to

1. have python  

2. have the code and set up the path so that python can find it  

3. have access to data  

4. code your feature and run it  

This should take no more than 10minutes altogether.  

Let's start:  

1. in a terminal check if you have python:  
```$which python```  
If this command returns an empty line, install python:   
```$sudo apt-get install python```  
---You might probably need to install a few libraries---  
A good alternative could be to install Anaconda as described [here](http://www.cs.ucl.ac.uk/scipython/help.html)

2. the code is hosted on a git repository on github.  
In short, a git repository is like a folder with all the history of changes stored and super useful tools to work collaboratively (merging contributions for ex)  
if you don't have git, install it:  
```$sudo apt-get install git```  
Then you'll need to create an account on [github](www.github.com) (worth it!)  
once you have an account, send a mail to Shaun Dowling (shaun.dowling.13@ucl.ac.uk) containing your github login and ask him to add you to the repo:  
"Hey Shaun, I'm xxx, I'd like to have access to the gatsby hackathon repo. Could you add me in? this is my id :yyyyy, thanks in advance, Best wishes, xxx"  
Once added (only), you can browse into the [project](https://github.com/smdowl/gatsby-hackathon-seizure) on github.  

3. once this is done, download the code.  
To do so, navigate with the command line to a folder where you want to store your code and type:  
```$git clone  https://github.com/smdowl/gatsby-hackathon-seizure.git```  
You now have a local copy of the repo that looks like a folder _gatsby-hackathon-seizures_

4. tell python where the code is  
each time you open a (bash) terminal, the script ~/.bashrc is run.  
edit it (using vim or gedit) and add the following line:  
```export PYTHONPATH=$PYTHONPATH:path_to_repo/code/python```

5. Either download or note down the path to the data. On the gatsby server, you may find it at: ```/nfs/data3/kaggle_seizure/```



Everything is now set up for you to run and edit the code

### what you should know about the data

The data consists of intradural _eeg_ recordings in dogs and humans indexed as subject. For each subject part of the data is labeled (_training set_), the rest is to be labeled (_test set_).
Data has been recorded for during both _ictal_ and _interictal_ episods.  

For the training data, the following information is provided:

* **subject_type**: dog or human
* **subject_index**: an integer
* **sampling_rate**: the sampling rate of acquisition
* **episode_type**: ictal or interictal
* **episode_index**: index of the ongoing recorded episode
* **latency**: each episode has a start time, latency is the elapsed time in second since the start time.

For the test data, the following information is provided:

* **subject_type**: dog or human
* **subject_index**: an integer
* **sampling_rate**: the sampling rate of acquisition
* **episode_type**: _NOT GIVEN_
* **episode_index**: _NOT GIVEN_
* **latency**: _NOT GIVEN_

> **Important note:** The training data has been stitched together per episode, in order of latency. This is both for visualization purpose and to simplify data manipulation.

#### Data organization.

Data is organized in folders named after subject type and index (ex: Dog_1). Within each folder, stitched trained episodes (ex: Dog_1_interictal_segment_1.mat) and unstiched individual test segments (ex: Dog_1_test_segment_234.mat)


### the minimum you need to know about the code

If you really want to focus on the science, then you just want to know how to write your features, how to choose/train your classifier and how to assess performance.  

The python code is here:  
_gatsby-hackathon-seizures/code/python/_  
Here you have a folder named seizure, containing different modules

* data:  to load and parse the data

* features: to declare features

* prediction: to declare the decision rules

* submission: to train, test and produce final submission file.

* evaluation: to do cross-validation evaluation

### Starting with an example

To get started you can directly read and run the ```getting_started.py``` file in the _examples_ folder.  
You just need to declare the path where your repository lies and then to run the script: ```$python getting_start.py```.


## Glossary

* leave one out: if there are N data points forming set X={x_i}, for each x_i, train on X\{x_i}, test on x_i, average prediction accuracy. 


