import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom as dcm
import numpy as np
from init_weights import init_weights
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_channels, num_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = input_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(self.n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, self.num_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        #act_parm = {'act_name':'relu'}
        #act_parm = {'act_name': 'MPELU', 'alpha': 0.25, 'beta': 1, 'mode': 'channel-shared'}
        #act_parm = {'act_name': 'EPReLU', 'eps': 0.4, 'mode': 'channel-shared'}
        #act_parm = {'act_name': 'EELU', 'alpha': 0.25, 'beta': 1, 'eps': 0.8, 'mode': 'channel-shared'}
        #act_parm = {'act_name': 'ELU'}
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        #self.activation1 = activation_function(middle_channels,act_parm)
        #self.activation2 = activation_function(out_channels, act_parm)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        #out = self.activation2(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.deep_supervision = deep_supervision

        nb_filter = [32, 64, 128, 256, 512]

        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            #print(output4.shape)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            #output = self.sigmoid(output)
            return output


'''
    UNet 3+
'''
#nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

        # self.E1_to_D4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        # self.E1_to_D4_conv = nn.Conv2d(filters[0], self.Channel64, 3, padding=1)
        # self.h1_PT_hd4_bn = nn.BatchNorm2d(self.Channel64)
        # self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

class Encoder2Decoder(nn.Module):
    def __init__(self, pool_size=2, input_channels=64, output_channels=64, do_pooling=True):
        super(Encoder2Decoder,self).__init__()
        self.pool_size = pool_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.do_pooling = do_pooling

        self.maxpool = nn.MaxPool2d(kernel_size=self.pool_size,stride=self.pool_size,ceil_mode=True)

        self.conv2d = nn.Conv2d(self.input_channels,self.output_channels,kernel_size=3,padding=1)
        self.batchNorm2d = nn.BatchNorm2d(self.output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.do_pooling == True:
            x = self.maxpool(x)
            x = self.conv2d(x)
            x = self.batchNorm2d(x)
        else:
            x = self.conv2d(x)
            x = self.batchNorm2d(x)

        output = self.relu(x)
        
        return output

     

class UNet3Plus(nn.Module):
    def __init__(self, n_channels=3, num_classes=1, bilinear=True, feature_scale=4,
                 is_deconv=True, is_batchnorm=True):
        super(UNet3Plus, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.n_channels, filters[0], self.is_batchnorm) # 1 -> 64 channels
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # 512*512 -> 256*256

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm) # 64 -> 128 channels
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 256*256 -> 128*128

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm) # 128 -> 256 channels
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) # 128*128 -> 64*64

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm) # 256 -> 512 channels
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) # 64*64 -> 32*32

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm) # 32*32*512 -> 32*32*1024

        ## -------------Decoder--------------
        self.Channel64 = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.Channel64 * self.CatBlocks

        '''stage 4d'''
        #512->256->128->64->32

        # h1->512*512, hd4->64*64, Pooling 8 times
        self.E1_to_D4 = Encoder2Decoder(pool_size=8,input_channels=filters[0],output_channels=64)

        # h2->256*256, hd4->64*64, Pooling 4 times
        self.E2_to_D4 = Encoder2Decoder(pool_size=4,input_channels=filters[1],output_channels=64)

        # h3->128*128, hd4->64*64, Pooling 2 times
        self.E3_to_D4 = Encoder2Decoder(pool_size=2,input_channels=filters[2],output_channels=64)

        # h4-> 64*64, hd4->64*64
        self.E4_to_D4 = Encoder2Decoder(input_channels=filters[3],output_channels=64,do_pooling=False)

        # hd5->32*32, hd4->64*64, Upsample 2 times
        self.D5_to_D4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.D5_to_D4_conv = Encoder2Decoder(input_channels=filters[4],output_channels=64,do_pooling=False)

        # fusion
        self.D4_concat = Encoder2Decoder(input_channels=self.UpChannels, output_channels=self.UpChannels,do_pooling=False)

        '''stage 3d'''
        # h1->512*512, hd3->128*128, Pooling 4 times
        self.E1_to_D3 = Encoder2Decoder(pool_size=4,input_channels=filters[0],output_channels=64)

        # h2->256*256, hd3->128*128, Pooling 2 times
        self.E2_to_D3 = Encoder2Decoder(pool_size=2,input_channels=filters[1],output_channels=64)

        # h3->128*128, hd3->128*128, Concatenation
        self.E3_to_D3 = Encoder2Decoder(input_channels=filters[2],output_channels=64,do_pooling=False)

        # hd4->64*64, hd4->128*128, Upsample 2 times
        self.D4_to_D3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.D4_to_D3_conv = Encoder2Decoder(input_channels=self.UpChannels,output_channels=64,do_pooling=False)

        # hd5->32*32, hd4->128*128, Upsample 4 times
        self.D5_to_D3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.D5_to_D3_conv = Encoder2Decoder(input_channels=filters[4],output_channels=64,do_pooling=False)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.D3_concat = Encoder2Decoder(input_channels=self.UpChannels, output_channels=self.UpChannels,do_pooling=False)


        '''stage 2d '''
        # h1->512*512, hd2->256*256, Pooling 2 times
        self.E1_to_D2 = Encoder2Decoder(pool_size=2,input_channels=filters[0],output_channels=64)

        # h2->160*160, hd2->160*160, Concatenation
        self.E2_to_D2 = Encoder2Decoder(pool_size=2,input_channels=filters[1],output_channels=64,do_pooling=False)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.D3_to_D2 = nn.Upsample(scale_factor=2, mode='bilinear') 
        self.D3_to_D2_conv = Encoder2Decoder(input_channels=self.UpChannels,output_channels=64,do_pooling=False)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.D4_to_D2 = nn.Upsample(scale_factor=4, mode='bilinear') 
        self.D4_to_D2_conv = Encoder2Decoder(input_channels=self.UpChannels,output_channels=64,do_pooling=False)


        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.D5_to_D2 = nn.Upsample(scale_factor=8, mode='bilinear') 
        self.D5_to_D2_conv = Encoder2Decoder(input_channels=filters[4],output_channels=64,do_pooling=False)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.D2_concat = Encoder2Decoder(input_channels=self.UpChannels, output_channels=self.UpChannels,do_pooling=False)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.E1_to_D1 = Encoder2Decoder(pool_size=2,input_channels=filters[0],output_channels=64)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.D2_to_D1 = nn.Upsample(scale_factor=2, mode='bilinear')  
        self.D2_to_D1_conv = Encoder2Decoder(input_channels=self.UpChannels,output_channels=64,do_pooling=False)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.D3_to_D1 = nn.Upsample(scale_factor=4, mode='bilinear')  
        self.D3_to_D1_conv = Encoder2Decoder(input_channels=self.UpChannels,output_channels=64,do_pooling=False)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.D4_to_D1 = nn.Upsample(scale_factor=8, mode='bilinear')  
        self.D4_to_D1_conv = Encoder2Decoder(input_channels=self.UpChannels,output_channels=64,do_pooling=False)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.D2_to_D1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.D2_to_D1_conv = Encoder2Decoder(input_channels=filters[4],output_channels=64,do_pooling=False)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.D1_concat = Encoder2Decoder(input_channels=self.UpChannels, output_channels=self.UpChannels,do_pooling=False)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, num_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        ## -------------Encoder-------------
        e1 = self.conv1(inputs)  # h1->512*512*64

        e2 = self.maxpool1(e1)
        e2 = self.conv2(e2)  # h2->160*160*128
        e3 = self.maxpool2(e2)
        e3 = self.conv3(e3)  # h3->80*80*256

        e4 = self.maxpool3(e3)
        e4 = self.conv4(e4)  # h4->40*40*512

        e5 = self.maxpool4(e4)
        d5 = self.conv5(e5)  # h5->20*20*1024

  
        ## -------------Decoder-------------
        e1_to_d4 = self.E1_to_D4(e1)
        e2_to_d4 = self.E2_to_D4(e2)
        e3_to_d4 = self.E3_to_D4(e3)
        e4_to_d4 = self.E4_to_D4(e4)
        d5_to_d4 = self.D5_to_D4_conv(self.D5_to_D4(d5))
        d4 = self.D4_concat(torch.concat((e1_to_d4,e2_to_d4,e3_to_d4,e4_to_d4,d5_to_d4),1))  # hd4->40*40*UpChannels

        e1_to_d3 = self.E1_to_D3(e1)
        e2_to_d3 = self.E2_to_D3(e2)
        e3_to_d3 = self.E3_to_D3(e3)
        d4_to_d3 = self.D4_to_D3_conv(self.D4_to_D3(d4))
        d5_to_d3 = self.D5_to_D3_conv(self.D5_to_D3(d5))
        d3 = self.D4_concat(torch.concat((e1_to_d3,e2_to_d3,e3_to_d3,d4_to_d3,d5_to_d3),1))


        e1_to_d2 = self.E1_to_D2(e1)
        e2_to_d2 = self.E2_to_D2(e2)
        d3_to_d2 = self.D3_to_D2_conv(self.D3_to_D2(d3))
        d4_to_d2 = self.D4_to_D2_conv(self.D4_to_D2(d4))
        d5_to_d2 = self.D5_to_D2_conv(self.D5_to_D2(d5))
        d2 = self.D4_concat(torch.concat((e1_to_d2,e2_to_d2,d3_to_d2,d4_to_d2,d5_to_d2),1))

        e1_to_d1 = self.E1_to_D1(e1)
        d2_to_d1 = self.D2_to_D1_conv(self.D2_to_D1(d2))
        d3_to_d1 = self.D3_to_D2_conv(self.D3_to_D2(d3))
        d4_to_d1 = self.D4_to_D2_conv(self.D4_to_D2(d4))
        d5_to_d1 = self.D5_to_D2_conv(self.D5_to_D2(d5))


        

        d1 = self.D4_concat(torch.concat((e1_to_d1,d2_to_d1,d3_to_d1,d4_to_d1,d5_to_d1),1))
        
      
        d1 = self.outconv1(d1)  


        return d1
    

