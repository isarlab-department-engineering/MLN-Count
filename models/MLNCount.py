import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.BaseModel import BaseModel
from models.LTPABlocks import LinearAttentionBlock


def predict_conv(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 8, kernel_size=3, padding=1),
        # nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

def final(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        # nn.ReLU(inplace=True)
    )

def crop_like(input, ref):
    assert (input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class MLNCountModel(BaseModel):

    def name(self):
        return 'UnetPACModel'

    def initialize(self, opt):
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['PALoss']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # self.visual_names = ['input', 'map_visual']
        # self.map_visual = None

        # self.histograms_names = []

        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['resnet50',
                            'upconv6', 'upconv5', 'upconv4', 'upconv3', 'upconv2', 'upconv1',
                            'iconv6', 'iconv5', 'iconv4', 'iconv3', 'iconv2', 'iconv1',
                            'predict_map4', 'predict_map3', 'predict_map2', 'predict_map1',
                            'final2', 'final1', 'atn3', 'atn2', 'atn1', 'classify'
                            ]

        self.netresnet50 = models.resnet50(pretrained=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        upconv_planes = [2048, 512, 256, 128, 64, 32, 16]

        self.netupconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.netupconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.netupconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.netupconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.netupconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.netupconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.neticonv6 = conv(upconv_planes[1] + 1024, upconv_planes[1])
        self.neticonv5 = conv(upconv_planes[2] + 512, upconv_planes[2])
        self.neticonv4 = conv(upconv_planes[3] + 256, upconv_planes[3])
        self.neticonv3 = conv(8 + upconv_planes[4] + 64, upconv_planes[4])
        self.neticonv2 = conv(8 + upconv_planes[5] + 64, upconv_planes[5])
        self.neticonv1 = conv(8 + upconv_planes[6], upconv_planes[6])

        self.netpredict_map4 = predict_conv(upconv_planes[3])
        self.netpredict_map3 = predict_conv(upconv_planes[4])
        self.netpredict_map2 = predict_conv(upconv_planes[5])
        self.netpredict_map1 = predict_conv(upconv_planes[6])

        self.netfinal2 = final(8, 8)
        self.netfinal1 = final(8, 8)

        self.netatn3 = LinearAttentionBlock(in_features=8, normalize_attn=True)
        self.netatn2 = LinearAttentionBlock(in_features=8, normalize_attn=True)
        self.netatn1 = LinearAttentionBlock(in_features=8, normalize_attn=True)

        self.netclassify =  nn.Linear(in_features=8 * 3, out_features=1, bias=True)

        if self.opt['is_train']:

            self.criterion = nn.BCEWithLogitsLoss()

            # initialize optimizers
            self.optimizer_resnet50 = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                             self.netresnet50.parameters()),
                                                      lr=opt['init_lr'] * 0.1)

            self.optimizers_upconvs = torch.optim.SGD(list(self.netupconv6.parameters()) +
                                                      list(self.netupconv5.parameters()) +
                                                      list(self.netupconv4.parameters()) +
                                                      list(self.netupconv3.parameters()) +
                                                      list(self.netupconv2.parameters()) +
                                                      list(self.netupconv1.parameters()), lr=opt['init_lr'])

            self.optimizers_iconvs = torch.optim.SGD(list(self.neticonv6.parameters()) +
                                                     list(self.neticonv5.parameters()) +
                                                     list(self.neticonv4.parameters()) +
                                                     list(self.neticonv3.parameters()) +
                                                     list(self.neticonv2.parameters()) +
                                                     list(self.neticonv1.parameters()), lr=opt['init_lr'])

            self.optimizers_map_predict = torch.optim.SGD(list(self.netpredict_map4.parameters()) +
                                                          list(self.netpredict_map3.parameters()) +
                                                          list(self.netpredict_map2.parameters()) +
                                                          list(self.netpredict_map1.parameters()), lr=opt['init_lr'])

            self.optimizers_final = torch.optim.SGD(list(self.netfinal1.parameters()) +
                                                    list(self.netfinal2.parameters()), lr=opt['init_lr'])

            self.optimizers_atn = torch.optim.SGD(list(self.netatn1.parameters()) +
                                                  list(self.netatn2.parameters()) +
                                                  list(self.netatn3.parameters()), lr=opt['init_lr'])

            self.optimizers_class = torch.optim.SGD(list(self.netclassify.parameters()), lr=opt['init_lr'])

            self.optimizers['resnet50'] = self.optimizer_resnet50
            self.optimizers['UpConvs'] = self.optimizers_upconvs
            self.optimizers['IConvs'] = self.optimizers_iconvs
            self.optimizers['MapPredicts'] = self.optimizers_map_predict
            self.optimizers['Final'] = self.optimizers_final
            self.optimizers['Atn'] = self.optimizers_atn
            self.optimizers['Class'] = self.optimizers_class

        if self.opt['gpu_id'] >= 0 and torch.cuda.is_available():
            self.netresnet50.cuda()

            self.netupconv6.cuda()
            self.netupconv5.cuda()
            self.netupconv4.cuda()
            self.netupconv3.cuda()
            self.netupconv2.cuda()
            self.netupconv1.cuda()

            self.neticonv6.cuda()
            self.neticonv5.cuda()
            self.neticonv4.cuda()
            self.neticonv3.cuda()
            self.neticonv2.cuda()
            self.neticonv1.cuda()

            self.netpredict_map4.cuda()
            self.netpredict_map3.cuda()
            self.netpredict_map2.cuda()
            self.netpredict_map1.cuda()

            self.netfinal1.cuda()
            self.netfinal2.cuda()
            self.netatn3.cuda()
            self.netatn2.cuda()
            self.netatn1.cuda()
            self.netclassify.cuda()

    def set_input(self, inputs):
        self.input = inputs[0]
        self.labels = inputs[1]
        if self.opt['gpu_id'] >= 0 and torch.cuda.is_available():
            self.input = self.input.cuda()
            self.labels = self.labels.cuda()


    def backward(self, retain_graph=False):

        loss = self.criterion(self.prediction, self.labels)
        loss.backward(retain_graph=retain_graph)
        self.loss_PALoss = loss

    def val_loss(self):

        loss = self.criterion(self.prediction, self.labels)
        self.loss_PALoss = loss

    # @staticmethod
    def forward(self):

        out_conv1 = self.netresnet50.conv1(self.input)
        out_conv1 = self.netresnet50.bn1(out_conv1)
        out_conv1 = self.netresnet50.relu(out_conv1)
        out_conv2 = self.netresnet50.maxpool(out_conv1)
        out_conv3 = self.netresnet50.layer1(out_conv2)

        # pooling between first and second resnet block needed to predict output at 4 scales
        out_conv3 = self.max_pool(out_conv3)
        out_conv4 = self.netresnet50.layer2(out_conv3)
        out_conv5 = self.netresnet50.layer3(out_conv4)
        out_conv6 = self.netresnet50.layer4(out_conv5)

        out_upconv6 = crop_like(self.netupconv6(out_conv6), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.neticonv6(concat6)

        out_upconv5 = crop_like(self.netupconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.neticonv5(concat5)

        out_upconv4 = crop_like(self.netupconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.neticonv4(concat4)
        disp4 = self.netpredict_map4(out_iconv4)

        out_upconv3 = crop_like(self.netupconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.neticonv3(concat3)
        disp3 = self.netpredict_map3(out_iconv3)

        out_upconv2 = crop_like(self.netupconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.neticonv2(concat2)
        disp2 = self.netpredict_map2(out_iconv2)

        out_upconv1 = crop_like(self.netupconv1(out_iconv2), self.input)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), self.input)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.neticonv1(concat1)
        disp1 = self.netpredict_map1(out_iconv1)

        out = self.netfinal2(disp1)

        c3, g3 = self.netatn3(disp3, crop_like(F.max_pool2d(out, kernel_size=4, stride=4, padding=2), disp3))
        c2, g2 = self.netatn2(disp2, crop_like(F.max_pool2d(out, kernel_size=2, stride=2, padding=1), disp2))
        c1, g1 = self.netatn1(disp1, out)

        g = torch.cat((g3, g2, g1), dim=1)
        self.prediction = self.netclassify(g)

        return c1 * out

    def forward_upsample(self):
        return self.forward()

    def inference_forward(self):
        with torch.no_grad():
            return self.forward()

    def optimize_parameters(self, epoch):
        # forward
        print('Optimizing')
        self.set_requires_grad([self.netresnet50,
                                self.netupconv6, self.netupconv5, self.netupconv4, self.netupconv3, self.netupconv2,
                                self.netupconv1,
                                self.neticonv6, self.neticonv5, self.neticonv4, self.neticonv3, self.neticonv2,
                                self.neticonv1,
                                self.netpredict_map4, self.netpredict_map3, self.netpredict_map2, self.netpredict_map1,
                                self.netfinal2, self.netfinal1, self.netatn3, self.netatn2, self.netatn1, self.netclassify
                                ], True)

        self.optimizer_resnet50.zero_grad()
        self.optimizers_upconvs.zero_grad()
        self.optimizers_iconvs.zero_grad()
        self.optimizers_map_predict.zero_grad()
        self.optimizers_final.zero_grad()
        self.optimizers_atn.zero_grad()
        self.optimizers_class.zero_grad()

        self.forward()
        self.backward(retain_graph=True)

        self.optimizer_resnet50.step()
        self.optimizers_upconvs.step()
        self.optimizers_iconvs.step()
        self.optimizers_map_predict.step()
        self.optimizers_final.step()
        self.optimizers_atn.step()
        self.optimizers_class.step()
