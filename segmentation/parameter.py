import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='parsenet', choices=['parsenet'])
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--version', type=str, default='parsenet')

    # Testing setting
    parser.add_argument('--test_size', type=int, default=2824) 
    parser.add_argument('--model_name', type=str, default='model.pth') 

    # Using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--img_path', type=str, default='./images')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--test_image_path', type=str, default='./images') 
    parser.add_argument('--test_label_path', type=str, default='./test_results') 
    parser.add_argument('--test_color_label_path', type=str, default='./test_color_visualize') 

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)

    # 未定義の引数を無視し、警告を表示
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignored unknown arguments: {unknown}")

    return args