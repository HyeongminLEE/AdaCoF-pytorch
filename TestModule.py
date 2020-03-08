from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
import os
from utility import to_variable


class Middlebury_eval:
    def __init__(self, input_dir='./evaluation'):
        self.im_list = ['Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Mequon', 'Schefflera', 'Teddy', 'Urban']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame11.png')).unsqueeze(0)))

    def Test(self, model, output_dir='./evaluation/output', output_name='frame10i11.png'):
        model.eval()
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))


class Middlebury_other:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', current_epoch) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


class Davis:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['bike-trial', 'boxing', 'burnout', 'choreography', 'demolition', 'dive-in', 'dolphins', 'e-bike', 'grass-chopper', 'hurdles', 'inflatable', 'juggle', 'kart-turn', 'kids-turning', 'lions', 'mbike-santa', 'monkeys', 'ocean-birds', 'pole-vault', 'running', 'selfie', 'skydive', 'speed-skating', 'swing-boy', 'tackle', 'turtle', 'varanus-tree', 'vietnam', 'wings-turn']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.get_epoch()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


class ucf:
    def __init__(self, input_dir):
        self.im_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame0.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame2.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame1.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.get_epoch()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)
