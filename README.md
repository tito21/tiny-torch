# Tiny-Torch

Many people use deep learning frameworks such as `pytorch` or `tensorflow` but few people can understand how they work
fully. And that is not a bad thing, these are complex pieces of software that took many years to perfect.

This project aims to make a simple deep learning framework that has all the main features of the major ones with a easy
to understand codebase.

The main features it must have
 - Fully functional compute engine with cpu and cuda backends
 - Automatic diferentation
 - Simple `pytorch` like python API for deep learning

# TODO

 - [ ] Parallelize matmul on cuda
 - [ ] Write nn module
 - [ ] Add examples
 - [ ] Improve how the backends are selected on the implementation

# Difference with pytorch

 - All tensors **always** requiere grads
 - the idioms `.cpu()` and `.cuda()` don't act in place you have to save the return tenor to a variable

# Building and installing

Just run `pip install .`

# Run test against pytorch

`pytest`

