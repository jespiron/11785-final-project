# Model Naming Scheme

Models in the main repo are named `x1-x2...-xn`, where `n` is the number of ancestors the model has including itself, and each `xi` indicates the path taken from the root ancestor.

n = 1 are models that are trained from scratch. Their names are `1`, `2`, `3`, etc.

n > 1 are models that descended from some other model in the main repo. Their names are `nameofparent-xn`, where `xn` indicates that it was the `xn`th child of parent. For example, `1-1` is the first child of model 1, `5-8-12` is the 12th child of model `5-8`, and so on.

# ⭐ Creating and Contributing Models

You can either train a model from scratch, or train from an existing model in this main repo.

## Training from scratch

**Model Creation**

1. Create a copy of `template/`
2. Rename the folder to `name-of-your-model/`. All naming power to you, no need to follow the model naming scheme
3. Train the model

**Contributing to Main Repo**

1. Create a copy of `name-of-your-model/`
2. Rename the folder according to the model naming scheme, by counting number of models trained from scratch in the main repo.
3. Open PR

## Training from existing model

1. Create a copy of the source model, which will be named of the form `name-of-source-model/`
2. Rename the folder according to the model naming scheme, which will be `name-of-source-model-xx/`, where `xx` is the number of children the source model will have upon creating your model.
3. Train the model

**Contributing to Main Repo**

1. Rename your model if you notice a name collision with the main repo; that is, someone else merged a child of your parent model before you did.
2. Open PR

# Training the Model

**Network Architecture**

* `network.py` to modify network architecture
* `data_set.py` to modify the way data is loaded and formatted for the network

**Training and Validation**

```python3 train.py```

```python3 validate.py```

Loads the model from your latest epoch unless specified otherwise.

After each epoch, your training and validation accuracies are printed. If you see an increase in training accuracy but decrease in validation accuracy, this is a sign of overfitting.

If you want to modify the meat of the training and eval logic, make your changes to `common.py`. If you think this is especially helpful, you can make these changes in `template/` and open a PR in the main repo.