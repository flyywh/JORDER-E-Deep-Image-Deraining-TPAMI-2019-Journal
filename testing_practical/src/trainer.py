import os
import math
from decimal import Decimal
import utility

import IPython
import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm

def grad_loss(img, gt):
    mse = nn.MSELoss(size_average=True)
    img_x = img[:,:,:,1:] - img[:,:,:,:-1]
    img_y = img[:,:,1:,:] - img[:,:,:-1,:]
    gt_x = gt[:,:,:,1:] - gt[:,:,:,:-1]
    gt_y = gt[:,:,1:,:] - gt[:,:,:-1,:]

    gt_x[gt_x.abs()<0.05] = 0
    gt_y[gt_y.abs()<0.05] = 0

    K_x = 1+3*torch.exp(-torch.abs(gt_x)/0.1)
    K_y = 1+3*torch.exp(-torch.abs(gt_y)/0.1)

    gt_x_enhance = gt_x*K_x
    gt_y_enhance = gt_y*K_y

    return mse(img_x, gt_x_enhance) + mse(img_y, gt_y_enhance)

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        loss1_all = 0
        loss2_all = 0
        loss3_all = 0
        cnt = 0

        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            cnt = cnt+1

            import numpy as np
#            if np.random.randint(10, size= (1, 1))<3:
#                lr = hr

            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr, sr2, mask, level = self.model(lr, idx_scale)
 
            self.mask_loss = torch.nn.CrossEntropyLoss(reduce=False, size_average=True)

            w1 = 10e-4 # * math.pow(0.5, int(epoch/2))
            w2 = 10e-3 # * math.pow(0.5, int(epoch/2))

            per_pixel_detection_loss = self.mask_loss(mask, ((hr-lr)[:,0,:,:]>0).type(torch.cuda.LongTensor))
            per_pixel_detection_loss = per_pixel_detection_loss.sum()

            loss1 = self.loss(sr, hr) + self.loss(sr2, hr) + 0.1 * grad_loss(sr2, hr)

            loss2 = w1*per_pixel_detection_loss
            loss3 = w2*self.loss(level, hr-lr)

            loss1_all = loss1_all + loss1
            loss2_all = loss2_all + loss2
            loss3_all = loss3_all + loss3

            loss = loss1 + loss2 + loss3

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        print(loss1_all / cnt)
        print(loss2_all / cnt)
        print(loss3_all / cnt)

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr, sr2, mask, level = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, epoch)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
