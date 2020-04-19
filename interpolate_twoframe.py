import argparse
from PIL import Image
import torch
from torchvision import transforms
import models
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Two-frame Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='adacofnet')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--config', type=str, default='./checkpoint/kernelsize_5/config.txt')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

parser.add_argument('--first_frame', type=str, default='./sample_twoframe/0.png')
parser.add_argument('--second_frame', type=str, default='./sample_twoframe/1.png')
parser.add_argument('--output_frame', type=str, default='./output.png')

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    config_file = open(args.config, 'r')
    while True:
        line = config_file.readline()
        if not line:
            break
        if line.find(':') == '0':
            continue
        else:
            tmp_list = line.split(': ')
            if tmp_list[0] == 'kernel_size':
                args.kernel_size = int(tmp_list[1])
            if tmp_list[0] == 'flow_num':
                args.flow_num = int(tmp_list[1])
            if tmp_list[0] == 'dilation':
                args.dilation = int(tmp_list[1])
    config_file.close()

    model = models.Model(args)

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])

    frame_name1 = args.first_frame
    frame_name2 = args.second_frame

    frame1 = to_variable(transform(Image.open(frame_name1)).unsqueeze(0))
    frame2 = to_variable(transform(Image.open(frame_name2)).unsqueeze(0))

    model.eval()
    frame_out = model(frame1, frame2)
    imwrite(frame_out.clone(), args.output_frame, range=(0, 1))


if __name__ == "__main__":
    main()
