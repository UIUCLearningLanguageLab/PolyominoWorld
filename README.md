# Polyomino World

## About

This repository contains research code for exploring the capabilities of simple neural networks to learn structured information about visual objects.

## Usage

### Getting Started

The code is designed to run on multiple machines, at the UIUC Learning & Language Lab using a custom job submission system called [Ludwig](https://github.com/phueb/Ludwig).
If you have access to the lab's file server, you can submit jobs with `Ludwig`.

To install `Ludwig` and all required dependencies, create a new virtual Python environment, then

```bash
 pip install -r requirements.txt
```

Note: The machines that will execute jobs may not have the same version of `torch` or `numpy`.
Check that you are using the same versions used by Ludwig workers [here](https://github.com/phueb/Ludwig#worker-specs).    

### Running jobs

To define which jobs will be executed, edit `polyomino_world/params.py`. To run each job 10 times,

```bash
ludwig -r10
```

Alternatively, jobs can be run locally:

```bash
ludwig --local
``` 

### Visualizing results

Clone [LudwigViz](https://github.com/phueb/Ludwig-Viz), navigate to the root directory, then

```bash
python -m flask run
```    

## Changelog

### February 21, 2021
- integrate with `Ludwig==3.0.0` job submission system.
- consolidated training and data creation scripts into a single function, `job.main`.
- simplify evaluation of models by saving performance data with `Ludwig`, and making names of performance curves more accessible.
- pass `world.World` data directly to `dataset.DataSet` instead of writing to and reading from disk.
- isolate instances of `shapes.Shape` from `world.World`.
- simplify generation of world data by explicitly generating `Sequences` with a single shape. Generation of `Sequence`s involving multiple shapes were never implemented directly.
- move logic for computing `wold_cells` in `shapes.Shape` to `world.World`.
- add dataclasses in `helpers.py` to enforce consistent attribute access and naming, and simplify attribute access.
- manipulate paths with `pathlib` instead of `os`.
- use `raise` instead of `sys.exit()` to exit program.
- use Python3.7 string formatting, e.g. `learning rate={}`.format(params.lr) -> f`learning rate={params.lr}`
- import module-levels objects instead of a module, e.g. `from polyominoworld import herlpers` -> `from polyominoworld.helpers import Action`.
- shorten variable names : e.g. `feature_list` -> `features`.
- add all instance initialization logic to `__init__` to make it easier to find the logic responsible for populating previously empty/placeholder attributes.
- remove duplicated code in `shapes.py`.
- remove any custom logic for writing, and reading data in `world.py`, `network_ml.py`, `network_sl.py`, and `evaluate.py`.
- use saving and loading functions provided by `torch` to save a mode's data.
- remove option to generate shapes in random world locations to temporarily consolidate logic for generating `Sequence`s.
- add `README.md`, and `requirements.txt`.
- remove outdated `notes.txt`.
- add Python3 type hints to function arguments.
- add brief summaries to the start of most Python files.
- clearly demarcate library code from scripting code by moving scripts into `scripts`.
- remove subdirectories from source code folder to simplify navigation.
- remove unused `__init__.py` files.
- remove many redundant or unnecessary `dict` and `list` objects to reduce chances of bug due to mismatch in data that is available in multiple objects.
- implemented option to train on `gpu`.
- remove unnecessary `.float()` calls on `torch` objects.
- rename 'HiddenState' to 'hidden', 'WorldState' to 'world', and 'FeatureVector' to 'features' to match lower-cased strings used everywhere else.
- use binary cross-entropy instead of MSE loss when `y_type='world`
- prepend `_` to method names to indicate they are used to perform "private" logic, and to distinguish them from methods which may be called externally.
- add doc strings to functions
- remove `shape.id_number` and populate active world cells with `shape.color` instead which can be directly used when creating `WorldVector`.
- remove `detailed_accuracies` and `accuracies` temporarily, to make evaluation logic more development-friendly.
- consolidate single and multi-layer networks into a single network capable of both.

## Compatibility

Developed on Ubuntu 18.04 with Python 3.7.9