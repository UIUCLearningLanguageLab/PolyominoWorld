"""
Convert pytorch model binary files to onnx format, 
import the onnx model to Tensorflow, 
save the TF file for the web visualization
"""

from pathlib import Path
import torch
import yaml


import onnx
from onnx_tf.backend import prepare
from polyominoworld.helpers import WorldVector
from polyominoworld.params import Params
from polyominoworld.params import param2requests, param2default
from polyominoworld.network import Network


from ludwig.results import gen_param_paths


def main():
    project_name = 'PolyominoWorld'
    for param_path, label in gen_param_paths(
            project_name,
            param2requests,
            param2default,
            runs_path=None,
            ludwig_data_path=None,
    ):

        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        for rep_id, path_to_net in enumerate(param_path.rglob('model.pt')):

            pytorch_model = Network(params)
            pytorch_model.load_state_dict(torch.load(path_to_net, map_location=torch.device('cpu')))
            pytorch_model.eval()

            dummy_input = torch.zeros(1, WorldVector.calc_size()) # dummy input with shape  model expects as input
            onnx_path_out = Path(__file__).parent.parent / 'onnx_models' / f'{param2val["param_name"]}_{rep_id}.onnx'
            if not onnx_path_out.parent.exists():
                onnx_path_out.parent.mkdir()
            torch.onnx.export(pytorch_model,
                              dummy_input,
                              onnx_path_out,
                              verbose=True)
            
            model = onnx.load(onnx_path_out)
            tf_rep = prepare(model)
            tf_path_out = Path(__file__).parent.parent / 'tensorflow_models' / f'{param2val["param_name"]}_{rep_id}.pb'
            if not tf_path_out.parent.exists():
                tf_path_out.parent.mkdir()
            tf_rep.export_graph(tf_path_out)


if __name__ == '__main__':
    main()