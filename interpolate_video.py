import argparse
from PIL import Image
import torch
from torchvision import transforms
import models
import os
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Video Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='adacofnet')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--config', type=str, default='./checkpoint/kernelsize_5/config.txt')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

parser.add_argument('--index_from', type=int, default=0, help='when index starts from 1 or 0 or else')
parser.add_argument('--zpad', type=int, default=3, help='zero padding of frame name.')

parser.add_argument('--input_video', type=str, default='./sample_video')
parser.add_argument('--output_video', type=str, default='./interpolated_video')

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
        if line.find(':') == 0:
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

    print('Loading the model...')

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])

    base_dir = args.input_video

    if not os.path.exists(args.output_video):
        os.makedirs(args.output_video)

    frame_len = len([name for name in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, name))])

    for idx in range(frame_len - 1):
        idx += args.index_from
        print(idx, '/', frame_len - 1, end='\r')

        frame_name1 = base_dir + '/' + str(idx).zfill(args.zpad) + '.png'
        frame_name2 = base_dir + '/' + str(idx + 1).zfill(args.zpad) + '.png'

        frame1 = to_variable(transform(Image.open(frame_name1)).unsqueeze(0))
        frame2 = to_variable(transform(Image.open(frame_name2)).unsqueeze(0))

        model.eval()
        frame_out = model(frame1, frame2)

        # interpolate
        imwrite(frame1.clone(), args.output_video + '/' + str((idx - args.index_from) * 2 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))
        imwrite(frame_out.clone(), args.output_video + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))

    # last frame
    print(frame_len - 1, '/', frame_len - 1)
    frame_name_last = base_dir + '/' + str(frame_len + args.index_from - 1).zfill(args.zpad) + '.png'
    frame_last = to_variable(transform(Image.open(frame_name_last)).unsqueeze(0))
    imwrite(frame_last.clone(), args.output_video + '/' + str((frame_len - 1) * 2 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))


if __name__ == "__main__":
    main()
