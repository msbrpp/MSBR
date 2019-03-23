# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
sys.path.append("..")
from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import AverageMeter as AverageMeter
from utils.utils import save_checkpoint as save_checkpoint
from utils.utils import Config as Config
import cpm_model
import lsp_lspet_data
import Mytransforms
from MSBR import MSBR


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained_d', default='None',type=str,
                        dest='pretrained_d', help='the path of pretrained model for detector')
    parser.add_argument('--pretrained_f', default='None',type=str,
                        dest='pretrained_f', help='the path of pretrained model for flownet')
    parser.add_argument('--train_dir', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, type=str,
                        dest='val_dir', help='the path of val file')
    parser.add_argument('--detector_name', default='../ckpt/detector', type=str,
                        help='model name to save parameters')
    parser.add_argument('--flownet_name', default='../ckpt/flownet', type=str,
                        help='model name to save parameters')
    parser.add_argument('--n_epochs', default=200, type=int,
                        dest='n_epochs', help='number of epochs')
    parser.add_argument('--loss_sup_weight', default=1., type=float,
                        dest='loss_sup_weight', help='weight of supervision loss')
    parser.add_argument('--loss_cv_weight', default=0.5, type=float,
                        dest='loss_cv_weight', help='weight of crossview loss')
    parser.add_argument('--loss_f_weight', default=0.5, type=float,
                        dest='loss_f_weight', help='weight of flownet loss')
    # parser.add_argument('--print_out_freq', default=20, type=int,
    #                     dest='print_out_freq', help='number of epochs')
    return parser.parse_args()




def get_parameters(model, config, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': config.base_lr},
            {'params': lr_2, 'lr': config.base_lr * 2.},
            {'params': lr_4, 'lr': config.base_lr * 4.},
            {'params': lr_8, 'lr': config.base_lr * 8.}]

    return params, [1., 2., 4., 8.]

# def MODEL()


def train(model, args, train_loader):

    global e

    batch_time  = AverageMeter()
    data_time   = AverageMeter()
    losses_sup  = AverageMeter()
    losses_cv   = AverageMeter()
    losses_f    = AverageMeter()
    losses      = AverageMeter()
    

    end = time.time()
    iters = config.start_iters
    best_model = config.best_model

    heat_weight = 46 * 46 * 15 / 1.0


    model.detector.train()
    model.flownet.train()

    

    for i, (input_sup, 
            heatmap_sup_gt, 
            input_cv,
            heatmap_cv_gt,
            input_f_t,
            input_f_s) in enumerate(train_loader):

        data_time.update(time.time() - end)


        input_sup       = input_sup.cuda()
        heatmap_sup_gt  = heatmap_sup_gt.cuda()
        input_cv        = input_cv.cuda()
        heatmap_cv_gt   = heatmap_cv_gt.cuda()

        input_f_t       = input_f_t.cuda()
        input_f_s       = input_f_s.cuda()

        # form dict for input and supervision
        inputs_detector = { 'sup_in': input_sup, 
                            'sup_gt': heatmap_sup_gt, 
                            'cv_s_in':  input_cv_s,
                            'cv_t_in':  input_cv_t}
                            
        inputs_flownet  = { 'f_t_in': input_f_t, 
                            'f_s_in': input_f_s}

        model.foward_compute_losses(inputs_detector, inputs_flownet)



        # update losses
        losses_sup.update(model.loss_sup.data[0], input_sup.size(0))
        losses_cv.update(model.loss_cv.data[0], input_sup.size(0))
        losses_f.update(model.loss_f.data[0], input_sup.size(0))
        losses.update(model.loss.data[0], input_sup.size(0))
        

        # gradient back-propag
        model.train_()

        batch_time.update(time.time() - end)
        end = time.time()

        iters += 1
        if i % config.display == 0:
            print('Epoch: {3}/{4}\t'
                    'Train Iteration: {0}/{size_train_set}\t'
                    'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'
                    'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                i, config.display, learning_rate, e, args.n_epochs, batch_time=batch_time,
                data_time=data_time, loss=losses, size_train_set=len(train_loader)))

            print(time.strftime(
            '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',time.localtime()))


    return losses_sup, losses_f
 
def val(model, args, val_loader, criterion, config):
    global e
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(6)]
    end = time.time()
    iters = config.start_iters
    best_model = config.best_model

    heat_weight = 46 * 46 * 15 / 1.0

    # model.eval()
    model.detector.eval()
    model.flownet.eval()
    # model.eval()
    for j, (input, heatmap, centermap) in enumerate(val_loader):
        heatmap = heatmap.cuda(async=True)
        centermap = centermap.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        heatmap_var = torch.autograd.Variable(heatmap)
        centermap_var = torch.autograd.Variable(centermap)

        output = model(input_var, centermap_var)

        loss_ = [criterion(ht, heatmap_var) * heat_weight for ht in output]

        loss = 0.
        loss += l for l in loss_
        losses.update(loss.data[0], input.size(0))
        for cnt, l in enumerate(
                [loss1, loss2, loss3, loss4, loss5, loss6]):
            losses_list[cnt].update(l.data[0], input.size(0))


        if j % config.display == 0:
            print('Valepoch: {2}/{3}\t'
                    'Test Iteration: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                    'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                j, config.display, e, args.n_epochs, batch_time=batch_time,
                data_time=data_time, loss=losses))
            for cnt in range(0, 6):
                print('Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                        .format(cnt + 1, loss1=losses_list[cnt]))

            print(time.strftime(
                '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',
                time.localtime()))
            batch_time.reset()
            losses.reset()
            for cnt in range(6):
                losses_list[cnt].reset()
        




def main(args):
    
    # build train and val set
    train_dir = args.train_dir
    val_dir = args.val_dir

    config = Config(args.config)
    cudnn.benchmark = True

    # train
    train_loader = torch.utils.data.DataLoader(
        lsp_lspet_data.LSP_Data('lspet', train_dir, 8,
                Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(368),
                Mytransforms.RandomHorizontalFlip(),
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)
    # val
    if args.val_dir is not None and config.test_interval != 0:
        # val
        val_loader = torch.utils.data.DataLoader(
            lsp_lspet_data.LSP_Data('lsp', val_dir, 8,
                              Mytransforms.Compose([Mytransforms.TestResized(368),
                                                    ])),
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True)
    
    # build model
    model = MSBR(config=config, args=args, k=14, stages=config.stages)

    model.build_nets()


    return model, train_loader, val_loader


    


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse()
    model, train_loader, val_loader = main(args)

    if args.pretrained_d is not 'None' and args.val_dir is not None and config.test_interval != 0:
        val_loss_d, val_loss_f = val(model, args, val_loader, criterion, config)
    
    for e in range(args.n_epochs):
        global e
        train_loss_d, train_loss_f = train(model, args, train_loader)

        if args.val_dir is not None and config.test_interval != 0:
            with torch.no_grad():
                val_loss_d, val_loss_f = val(model, args, val_loader, criterion)


            is_best_d = val_loss_d.avg < config.best_model_d
            is_best_f = val_loss_f.avg < config.best_model_f
            config.best_model_d = min(config.best_model_d, losses.avg)
            config.best_model_f = min(config.best_model_f, losses.avg)
            save_checkpoint({
                'epoch': e,
                'state_dict': model.detector.state_dict(),
            }, is_best_d, args.detector_name)
            save_checkpoint({
                'epoch': e,
                'state_dict': model.flownet.state_dict(),
            }, is_best_f, args.flownet_name)
