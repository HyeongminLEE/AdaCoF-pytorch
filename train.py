from datareader import DBreader_Vimeo90k
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
import torch
from TestModule import Middlebury_other
import models
from trainer import Trainer
import losses
import datetime

parser = argparse.ArgumentParser(description='AdaCoF-Pytorch')

# parameters
# Model Selection
parser.add_argument('--model', type=str, default='adacofnet')

# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str, default='./db/vimeo_triplet')
parser.add_argument('--out_dir', type=str, default='./output_adacof_train')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--test_input', type=str, default='./test_input/middlebury_others/input')
parser.add_argument('--gt', type=str, default='./test_input/middlebury_others/gt')

# Learning Options
parser.add_argument('--epochs', type=int, default=50, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--loss', type=str, default='1*Charb+0.01*g_Spatial+0.005*g_Occlusion', help='loss function configuration')
parser.add_argument('--patch_size', type=int, default=256, help='Patch size')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Options for AdaCoF
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

transform = transforms.Compose([transforms.ToTensor()])


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    dataset = DBreader_Vimeo90k(args.train, random_crop=(args.patch_size, args.patch_size))
    TestDB = Middlebury_other(args.test_input, args.gt)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = models.Model(args)
    loss = losses.Loss(args)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    my_trainer = Trainer(args, train_loader, TestDB, model, loss, start_epoch)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(args.out_dir + '/config.txt', 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()

    my_trainer.close()


if __name__ == "__main__":
    main()
