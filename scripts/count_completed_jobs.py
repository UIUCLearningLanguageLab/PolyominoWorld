"""
Count completed jobs, by counting number of saved pytorch models, assuming each job saves 1 model file.

To look for results on the shared drive, we use ludwig.
We use params.param2requests to tell ludwig which jobs we would like results for.
To tell ludwig where to look for results,
create an environment variable "LUDWIG_MNT" that points to the path where ludwig_data is mounted on your machine.
"""

from ludwig.results import gen_param_paths


from polyominoworld.params import param2default, param2requests


num_completed = 0
project_name = 'PolyominoWorld'
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=None,
                                         ludwig_data_path=None,
                                         label_n=True):
    saved_models = [path for path in param_path.rglob('model.pt')]
    num_completed += len(saved_models)

print(f'Found {num_completed} completed jobs.')


