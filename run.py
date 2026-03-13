import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
from utils.configer import load_config
from CG_SLAM import CG_SLAM
from gaussian_utils.general_utils import safe_state
from gaussian_model.gaussian_arguments import OptimizationParams, ModelParams, PipelineParams


def resolve_device(device_arg):
    if device_arg is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CG-SLAM currently requires a CUDA-capable device. Please rerun on a CUDA-enabled machine.")

    device = torch.device(device_arg)
    if device.type != "cuda":
        raise RuntimeError("CG-SLAM currently supports CUDA devices only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
    return device


def main():
    '''
    Main entry of this project.
    '''
    parser = argparse.ArgumentParser(description= 'Arguments in CG_SLAM')

    hyper_dict = {'model':ModelParams(parser), 'optim': OptimizationParams(parser), 'pipe': PipelineParams(parser)}


    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--config', type = str, help = 'Path to config file', default='./configs/Replica/room0.yaml')
    parser.add_argument('-I', '--input_folder', type = str, help = 'Path to input data', default = None)
    parser.add_argument('-O', '--output_folder', type = str, help = 'Path to output data', default = None)
    parser.add_argument('--device', type=str, default=None, help='CUDA device used at runtime, e.g. cuda:0 or cuda:1.')
    parser.add_argument('--light', default=False, action='store_true', help='Enable light mode (use crop_size from config for half resolution).')

    args = parser.parse_args()
    args.save_iterations.append(args.iterations)
    args.device = str(resolve_device(args.device))

    # Initialize system state (RNG)
    safe_state(args.quiet, args.device)
    # autograd detection
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    hyper_group = {'model':hyper_dict['model'].extract(args), 'optim':hyper_dict['optim'].extract(args), 'pipe': hyper_dict['pipe'].extract(args)}
    
    cfg = load_config(args.config)
    
    cg_slam = CG_SLAM(cfg, args, hyper_group)

    cg_slam.run()
    
if __name__ == '__main__':
    main()
    print("SLAM Complete!!!")
