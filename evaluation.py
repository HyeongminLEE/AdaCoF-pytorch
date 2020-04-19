import argparse
import torch
import models
import os
import TestModule

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='adacofnet')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--config', type=str, default='./checkpoint/kernelsize_5/config.txt')
parser.add_argument('--out_dir', type=str, default='./output_adacof_test')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    if args.config is not None:
        config_file = open(args.config, 'r')
        while True:
            line = config_file.readline()
            if not line:
                break
            if line.find(':') == 0:
                continue
            else:
                tmp_list = line.split(': ')
                if tmp_list[0] == 'kernel_size':
                    args.kernel_size = int(tmp_list[1])
                if tmp_list[0] == 'dilation':
                    args.dilation = int(tmp_list[1])
        config_file.close()

    model = models.Model(args)

    print('Loading the model...')

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])
    current_epoch = checkpoint['epoch']

    print('Test: Middlebury_eval')
    test_dir = args.out_dir + '/middlebury_eval'
    test_db = TestModule.Middlebury_eval('./test_input/middlebury_eval')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir)

    print('Test: Middlebury_others')
    test_dir = args.out_dir + '/middlebury_others'
    test_db = TestModule.Middlebury_other('./test_input/middlebury_others/input', './test_input/middlebury_others/gt')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir, current_epoch, output_name='frame10i11.png')

    print('Test: DAVIS')
    test_dir = args.out_dir + '/davis'
    test_db = TestModule.Davis('./test_input/davis/input', './test_input/davis/gt')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir, output_name='frame10i11.png')

    print('Test: UCF101')
    test_dir = args.out_dir + '/ucf101'
    test_db = TestModule.ucf('./test_input/ucf101')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir, output_name='frame1.png')


if __name__ == "__main__":
    main()
