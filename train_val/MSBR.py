import torch.nn as nn
import torch.nn.functional as F
import torch
import cpm_model
from model.flownet2c import FlowNetC

def get_parameters(detector, config, isdefault=True):
    if isdefault:
        return detector.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(detector.module.named_parameters())
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

class MSBR:
    def __init__(self, config, args, k, stages):
        self.config = config
        self.args   = args
        self.k      = k
        self.stages = stages
        # self.training = 

    def build_nets(self):
        def build_detectors(self):
            self.detector = cpm_model.CPM(k=self.k, stages=self.stages)
            self.detector = torch.nn.DataParallel(self.detector, device_ids=self.args.gpu).cuda()

            # reload paras for pretrained self.detector is availble
            if self.args.pretrained_d != 'None':
                state_dict = torch.load(self.args.pretrained_d)['state_dict']
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():

                    name = k[7:]
                    new_state_dict[name] = v
                self.detector.load_state_dict(new_state_dict)
            pass
    
        def build_flow_estimator(self):
            # we have alternative choice of using flownet2 to estimate optical, however we do not use here.
            self.flownet = FlowNetC(batchNorm=True)
            self.flownet = torch.nn.DataParallel(self.flownet, device_ids=self.args.gpu).cuda()
            # reload paras for pretrained self.detector is availble
            if self.args.pretrained_f != 'None':
                state_dict = torch.load(self.args.pretrained_f)['state_dict']
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():

                    name = k[7:]
                    new_state_dict[name] = v
                self.flownet.load_state_dict(new_state_dict)
            pass

        
        # build two models/nets
        self.build_detectors()
        self.criterion_d = nn.MSELoss().cuda()

        self.build_flow_estimator()
        self.criterion_f = nn.MSELoss().cuda()

        # extract params from two models
        params_d, _ = get_parameters(self.detector, self.config, False)
        params_f, _ = get_parameters(self.flownet, self.config, False)
        assert type(params_d) is list and type(params_f) is list
        params      = params_d + params_f

        # setup optimizer
        self.optimizer = torch.optim.Adam(params, lr=self.config.base_lr)

    
    def foward_compute_losses(self, inputs_detector, inputs_flownet, training=True):
        def _computeL_sup(heatmaps_d, heatmap_gt, criterion):
            loss_ = [criterion(ht, heatmap_var) * heat_weight for ht in output]
            loss_sup = 0.
            loss_sup += l for l in loss_
            return loss_sup

        def _computeL_crossview(heatmaps_cv_s, heatmaps_cv_t, criterion):
            loss_ = criterion(heatmaps_cv_s[0], heatmaps_cv_t[0])

            return loss_
            
        
        def _computeL_f(input_f_s, input_f_t, flows, criterion):
            scales_f = len(flows)
            loss_f = 0.

            # input_f_s = self.input_f[:]
            # input_f_t = self.input_f[:]
            for s_f in range(scales_f):
                input_f_t_w = warping(flows[s_f], input_f_s)

                loss_f += criterion(input_f_t_w, input_f_t)
            return loss_f
        
        # extract feed-in
        input_sup = inputs_detector['sup_in']
        heatmaps_sup_gt = inputs_detector['sup_gt']

        input_cv_s = inputs_detector['cv_s_in']
        input_cv_t = inputs_detector['cv_t_in']

        input_f_t = inputs_flownet['f_t_in']
        input_f_s = inputs_flownet['f_s_in']
        
 
        # detector forward and loss
        ## supervision loss
        heatmaps_sup    = self.detector.forward(input_sup)
        self.loss_sup   = _computeL_sup(heatmaps_sup, heatmaps_sup_gt)

        ## crossview loss
        heatmaps_cv_s     = self.detector.forward(input_cv_s)
        heatmaps_cv_t     = self.detector.forward(input_cv_t)
        self.loss_cv      = _computeL_crossview(heatmaps_cv_s, heatmaps_cv_t)

        # flownet/flow estimator forward
        flows = self.flownet.forward(input_f)
        # self.loss_f = _computeL_f(input_f_s, input_f_t, flows)

        # total loss
        self.loss = self.w_sup * self.loss_sup +self.w_crossview * self.loss_crossview #+ self.w_f * self.loss_f
        


        pass
    
    def train_(self):
        self.foward_compute_losses()


        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        pass
    
    def eval_(self):
        with torch.no_grad():

        pass

