import os
import torch
import utility
from utility import to_variable


class Trainer:
    def __init__(self, args, train_loader, test_loader, my_model, my_loss, start_epoch=0):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.test_loader = test_loader
        self.model = my_model
        self.loss = my_loss
        self.current_epoch = start_epoch

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        self.result_dir = args.out_dir + '/result'
        self.ckpt_dir = args.out_dir + '/checkpoint'

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.logfile = open(args.out_dir + '/log.txt', 'w')

        # Initial Test
        self.model.eval()
        self.test_loader.Test(self.model, self.result_dir, self.current_epoch, self.logfile, str(self.current_epoch).zfill(3) + '.png')

    def train(self):
        # Train
        self.model.train()
        for batch_idx, (frame0, frame1, frame2) in enumerate(self.train_loader):
            frame0 = to_variable(frame0)
            frame1 = to_variable(frame1)
            frame2 = to_variable(frame2)

            self.optimizer.zero_grad()

            output = self.model(frame0, frame2)
            loss = self.loss(output, frame1, [frame0, frame2])
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ', '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ', '[' + str(batch_idx) + '/' + str(self.max_step) + ']', 'train loss: ', loss.item()))
        self.current_epoch += 1
        self.scheduler.step()

    def test(self):
        # Test
        torch.save({'epoch': self.current_epoch, 'state_dict': self.model.get_state_dict()}, self.ckpt_dir + '/model_epoch' + str(self.current_epoch).zfill(3) + '.pth')
        self.model.eval()
        self.test_loader.Test(self.model, self.result_dir, self.current_epoch, self.logfile, str(self.current_epoch).zfill(3) + '.png')
        self.logfile.write('\n')

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()
