"""
Use this script to insert novel parameters into existing param2val.yaml files saved to runs_path
"""
import yaml

from polyominoworld.params import get_runs_path

ADDED_PARAMS = {'shuffle_input': False}


runs_path = get_runs_path()

for param_path in sorted(runs_path.glob('param_*')):

    print(f'Modifying {param_path}/param2val.yaml')

    # load
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in ADDED_PARAMS.items():
        if k not in param2val:
            param2val[k] = v

    # write
    with (param_path / 'param2val.yaml').open('w', encoding='utf8') as f:
        yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)
