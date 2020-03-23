# Kickstarter

Kickstarter is a community of more than 10 million people comprising of creative, tech enthusiasts who help in bringing creative project to life.

Until now, more than $3 billion dollars have been contributed by the members in fueling creative projects.
The projects can be literally anything – a device, a game, an app, a film etc.

Kickstarter works on all or nothing basis i.e if a project doesn’t meet its goal, the project owner gets nothing.
For example: if a projects’s goal is $5000. Even if it gets funded till $4999, the project won’t be a success.

If you have a project that you would like to post on Kickstarter now, can you predict whether it will be successfully funded or not? Looking into the dataset, what useful information can you extract from it, which variables are informative for your prediction and can you interpret the model?

The goal of this project is to build a classifier to predict whether a project will be successfully funded, you can use the algorithm of your choice.

**Notes**:
* The target, `state` can take two values `successful` or `'failed'`). Here we want to convert it to a binary outcome: `1` if `successful` or `0` if `failed`.
* The variables `'deadline'', 'created_at', 'launched_at'` are stored in Unix time format.

## Get the data

We provide a file, `run.py` that you can use to manage the project. To download the data in the right place; in a terminal run:

```python
python run.py setup
```
from within the repository.

Alternatively, you can download the data manually and place it in a `data/` folder:
* Download the dataset from [here](https://s3-eu-west-1.amazonaws.com/kate-datasets/kickstarter/train.zip).
* Create a new directory called `data/`
* Place it in the `data/` folder.

Since the dataset is quite big, it will not be tracked by the system (see the `.gitignore` file (but don't change it)).


## Start working

You will need to implement the class `KickstarterModel` in `model.py` and use the `run.py` to train your model and save its state to a file.

Check the Machine Learning help page on K.A.T.E. for more details on how to do so.