# Decoupled Gaussian process model.

**This is not an officially supported Google product.**

This repository contains an implementation of the Decoupled Gaussian Process
model that decouples the representation of mean and covariance in reproducing
kernel Hilbert space.

The details of the model is in the paper: _Cheng, Ching-An, and Byron Boots.
"Variational Inference for Gaussian Process Models with Linear Complexity."
Advances in Neural Information Processing Systems. 2017._

Link to the paper:
http://papers.nips.cc/paper/7103-variational-inference-for-gaussian-process-models-with-linear-complexity

## How to use the model

This model can be used mainly in two ways:

*   through `session.run()` (detailed in
    `decoupled_gaussian_process_example.py`), and
*   through `tf.estimator.Estimator` with `model_fn()` (defined in
    `decoupled_gaussian_process_model.py`).

### Through `session.run()`

File `decoupled_gaussian_process_example.py` provides detailed steps to train
and evaluate the model by first building the graph, and then iteratively
minimizing the objective function by `session.run(train_step)`. In order to use
the model as a layer, you may want to embed the logic of adding bases online and
hyperparameters initialization in the graph, so that no initial values for
hyperparameters are needed and no need to call `model.bases.add_bases()` in the
train loop anymore.

### Through `model_fn()`

`model_fn()` is defined in `decoupled_gaussian_process_model.py`. We can use the
following code to create an `tf.estimator.Estimator`:

```python
estimator = tf.estimator.Estimator(
    model_fn=decoupled_gaussian_process_model.model_fn,
    model_dir=run_config.model_dir,
    params=hparams,
    config=run_config)
```

Then the `tf.estimator.Estimator` can be used with
`tf.contrib.learn.Experiment`, to quickly carry out experiments to compare
against other models.
