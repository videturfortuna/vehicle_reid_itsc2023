import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import copy


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MHA(nn.Module):
    def __init__(self, n_dims, heads=4):
        super(MHA, self).__init__()
        self.query = nn.Linear(n_dims, n_dims)
        self.key = nn.Linear(n_dims, n_dims)
        self.value = nn.Linear(n_dims, n_dims)

        self.mha = torch.nn.MultiheadAttention(n_dims, heads)
        print('debug')

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        out = self.mha(q,k,v)

        return out


## Transformer Block
##multi Head attetnion from BoTnet https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
class MHSA(nn.Module):
    def __init__(self, n_dims, width=16, height=16, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        ### , bias = False in conv2d
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size() # C // self.heads,
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, torch.div(C, self.heads, rounding_mode='floor'), -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

###also from https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py

class Bottleneck_Transformer(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, resolution=None, use_mlp = False):
        super(Bottleneck_Transformer, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList()
        self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
        if stride == 2:
            self.conv2.append(nn.AvgPool2d(2, 2))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.use_MLP = use_mlp
        if use_mlp:
            self.LayerNorm = torch.nn.InstanceNorm2d(in_planes)
            self.MLP_torch = torchvision.ops.MLP(in_planes, [512, 2048])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.use_MLP:
            residual = out
            out = self.LayerNorm(out)
            out = out.permute(0,3,2,1)
            out = self.MLP_torch(out)
            out = out.permute(0,3,2,1)
            out = out + residual
            # out = F.relu(out)
        return out





# Defines the new fc layer and classification layer
# |--MLP--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.0, relu=False, bnorm=True, linear=False, return_f = True, circle=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.circle = circle
        add_block = []
        if linear: ####MLP to reduce
            final_dim = linear
            add_block += [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, final_dim)]
        else:
            final_dim = input_dim
        if bnorm:
            tmp_block = nn.BatchNorm1d(final_dim)
            tmp_block.bias.requires_grad_(False) 
            add_block += [tmp_block]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(final_dim, class_num, bias=False)] # 
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        if x.dim()==4:
            x = x.squeeze().squeeze()
        if x.dim()==1:
            x = x.unsqueeze(0)
        x = self.add_block(x)
        if self.return_f:
            f = x
            if self.circle:
                x = F.normalize(x)
                self.classifier[0].weight.data = F.normalize(self.classifier[0].weight, dim=1)
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class GroupNormwMomentum(nn.Module):
    def __init__(self, n_groups, n_channels):
        super().__init__()
        
        self.gn = nn.ModuleList()
        for i in range(n_groups):
            self.gn.append(nn.BatchNorm2d(int(n_channels/2)).apply(weights_init_kaiming))
    
    def forward(self, x):
        size_feat = int(x.size(1)/2)
        x_left  = x[:,:size_feat]
        x_right = x[:,size_feat:]
        x_left = self.gn[0](x_left)
        x_right = self.gn[1](x_right)
        return torch.cat((x_left, x_right), dim=1)



class Conv_MHSA_2G(nn.Module):
    def __init__(self, c_in, c_out, resolution=[16,16], heads=4) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(c_in//2, c_out//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.MHSA_1 = MHSA(c_out//2, width=int(resolution[0]), height=int(resolution[1]), heads=heads)


    def forward(self,x):
        x_1 = self.conv2(x[:,:x.size(1)//2,:,:])
        x_2 = self.MHSA_1(x[:,x.size(1)//2:,:,:])

        x = torch.cat((x_1, x_2), dim=1)

        return x

class Conv_MHSA_4G(nn.Module):
    def __init__(self, c_in, c_out, resolution=[16,16], heads=4) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(c_in//2, c_out//2, kernel_size=3, stride=1, padding=1, groups=2, bias=False)
        self.MHSA_1 = MHSA(c_out//4, width=int(resolution[0]), height=int(resolution[1]), heads=heads)
        self.MHSA_2 = MHSA(c_out//4, width=int(resolution[0]), height=int(resolution[1]), heads=heads)

    def forward(self,x):
        x_12 = self.conv2(x[:,:x.size(1)//2,:,:])
        x_3 = self.MHSA_1(x[:,x.size(1)//2:x.size(1)//2+x.size(1)//4,:,:])
        x_4 = self.MHSA_2(x[:,x.size(1)//2+x.size(1)//4:,:,:])
        x = torch.cat((x_12, x_3, x_4), dim=1)

        return x
class MHSA_2G(nn.Module):
    def __init__(self, c_out, resolution=[16,16], heads=4) -> None:
        super().__init__()
        self.MHSA_1 = MHSA(int(c_out//2), width=int(resolution[0]), height=int(resolution[1]), heads=heads)
        self.MHSA_2 = MHSA((c_out//2), width=int(resolution[0]), height=int(resolution[1]), heads=heads)

    def forward(self,x):
        x_ce = self.MHSA_1(x[:,:x.size(1)//2,:,:])
        x_t = self.MHSA_2(x[:,x.size(1)//2:,:,:])
        x = torch.cat((x_ce, x_t), dim=1)

        return x
    



class base_branches(nn.Module):
    def __init__(self, backbone="ibn", stride=1):
        super(base_branches, self).__init__()
        if backbone == 'r50':
            model_ft = models.resnet50()
        elif backbone == '101ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        elif backbone == '34ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        else:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
            
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            if backbone == "34ibn":
                model_ft.layer4[0].conv1.stride = (1,1)
            else:
                model_ft.layer4[0].conv2.stride = (1,1)

        self.model = torch.nn.Sequential(*(list(model_ft.children())[:-3])) 

    def forward(self, x):
        x = self.model(x)
        return x
    
class multi_branches(nn.Module):
    def __init__(self, n_branches, n_groups, pretrain_ongroups=True, end_bot_g=False, group_conv_mhsa=False, group_conv_mhsa_2=False, x2g = False, x4g=False):
        super(multi_branches, self).__init__()

        model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        model_ft= model_ft.layer4
        self.x2g = x2g
        self.x4g = x4g
        if n_groups > 0:
            convlist = [k.split('.') for k, m in model_ft.named_modules(remove_duplicate=False) if isinstance(m, nn.Conv2d)]
            for item in convlist:
                if item[1] == "downsample":
                    m = model_ft[int(item[0])].get_submodule(item[1])[0]
                else:
                    m = model_ft[int(item[0])].get_submodule(item[1]) #'.'.join(
                weight = m.weight[:int(m.weight.size(0)), :int(m.weight.size(1)/n_groups), :,:]
                if end_bot_g and item[1]=="conv2":
                    setattr(model_ft[int(item[0])], item[1], MHSA_2G(int(512), int(512)))
                elif group_conv_mhsa and item[1]=="conv2":
                    setattr(model_ft[int(item[0])], item[1], Conv_MHSA_4G(int(512), int(512)))
                elif group_conv_mhsa_2 and item[1]=="conv2":
                    setattr(model_ft[int(item[0])], item[1], Conv_MHSA_2G(int(512), int(512)))
                else:
                    if item[1] == "downsample":
                        getattr(model_ft[int(item[0])], item[1])[0] = nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming)
                        if pretrain_ongroups:
                            getattr(model_ft[int(item[0])], item[1])[0].weight.data = weight
                    elif item[1] == "conv2":
                        setattr(model_ft[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=3, stride=1, padding=(1,1), groups=n_groups, bias=False).apply(weights_init_kaiming))
                        if pretrain_ongroups:
                            setattr(model_ft[int(item[0])].get_submodule(item[1]).weight, "data", weight)                        
                    else:
                        setattr(model_ft[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming))
                        if pretrain_ongroups:
                            setattr(model_ft[int(item[0])].get_submodule(item[1]).weight, "data", weight)
        self.model = nn.ModuleList()

        if len(n_branches) > 0:
            if n_branches[0] == "2x":
                self.model.append(model_ft)
                self.model.append(copy.deepcopy(model_ft))
            else:
                for item in n_branches:
                    if item =="R50":
                        self.model.append(copy.deepcopy(model_ft))
                    elif item == "BoT":
                        layer_0 = Bottleneck_Transformer(1024, 512, resolution=[16, 16], use_mlp = False)
                        layer_1 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = False)
                        layer_2 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = False)
                        self.model.append(nn.Sequential(layer_0, layer_1, layer_2))
                    else:
                        print("No valid architecture selected for branching by expansion!")
        else:
            self.model.append(model_ft)


    def forward(self, x):
        output = []
        for cnt, branch in enumerate(self.model):
            if self.x2g and cnt>0:
                aux = torch.cat((x[:,int(x.shape[1]/2):,:,:], x[:,:int(x.shape[1]/2),:,:]), dim=1)
                output.append(branch(aux))
            elif self.x4g and cnt>0:
                aux = torch.cat((x[:,int(x.shape[1]/4):int(x.shape[1]/4*2),:,:], x[:, :int(x.shape[1]/4),:,:], x[:, int(x.shape[1]/4*3):,:,:], x[:, int(x.shape[1]/4*2):int(x.shape[1]/4*3),:,:]), dim=1)
                output.append(branch(aux))
            else:
                output.append(branch(x))
       
        return output

class FinalLayer(nn.Module):
    def __init__(self, class_num, n_branches, n_groups, losses="LBS", droprate=0, linear_num=False, return_f = True, circle_softmax=False, n_cams=0, n_views=0, LAI=False, x2g=False,x4g=False):
        super(FinalLayer, self).__init__()    
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.finalblocks = nn.ModuleList()
        self.withLAI = LAI
        if n_groups > 0:
            self.n_groups = n_groups
            for i in range(n_groups*(len(n_branches)+1)):
                if losses == "LBS":
                    if i%2==0:
                        self.finalblocks.append(ClassBlock(int(2048/n_groups), class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
                    else:
                        bn= nn.BatchNorm1d(int(2048/n_groups))
                        bn.bias.requires_grad_(False)  
                        bn.apply(weights_init_kaiming)
                        self.finalblocks.append(bn)
                else:
                    self.finalblocks.append(ClassBlock(int(2048/n_groups), class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
        else:
            self.n_groups = 1
            for i in range(len(n_branches)):
                if losses == "LBS":
                    if i%2==0:
                        self.finalblocks.append(ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
                    else:
                        bn= nn.BatchNorm1d(int(2048))
                        bn.bias.requires_grad_(False)  
                        bn.apply(weights_init_kaiming)
                        self.finalblocks.append(bn)
                else:
                    self.finalblocks.append(ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))

        if losses == "LBS":
            self.LBS = True
        else:
            self.LBS = False

        if self.withLAI:
            # self.LAI = []
            self.n_cams = n_cams
            self.n_views = n_views
            if n_groups>0 and len(n_branches)==0:
                n_branches = ["groups"]
            if n_cams>0 and n_views>0:
                if x2g or x4g:
                    self.LAI = nn.Parameter(torch.zeros(2, n_cams * n_views, 2048))
                else:
                    self.LAI = nn.Parameter(torch.zeros(len(n_branches), n_cams * n_views, 2048))
            elif n_cams>0:
                if x2g or x4g:
                    self.LAI = nn.Parameter(torch.zeros(2, n_cams, 2048))
                else:
                    self.LAI = nn.Parameter(torch.zeros(len(n_branches), n_cams, 2048))
            elif n_views>0:
                if x2g or x4g:
                    self.LAI = nn.Parameter(torch.zeros(2, n_views, 2048))
                else:
                    self.LAI = nn.Parameter(torch.zeros(len(n_branches), n_views, 2048))
            else: self.withLAI = False

    def forward(self, x, cam, view):
        # if len(x) != len(self.finalblocks):
        #     print("Something is wrong")
        embs = []
        ffs = []
        preds = []
        for i in range(len(x)):
            emb = self.avg_pool(x[i]).squeeze(dim=-1).squeeze(dim=-1)
            if self.withLAI:
                if self.n_cams > 0 and self.n_views >0:
                    emb = emb + self.LAI[i, cam * self.n_views + view, :]
                elif self.n_cams >0:
                    emb = emb + self.LAI[i, cam, :]
                else:
                    emb = emb + self.LAI[i, view, :]
            for j in range(self.n_groups):
                aux_emb = emb[:,int(2048/self.n_groups*j):int(2048/self.n_groups*(j+1))]
                if self.LBS:
                    if (i+j)%2==0:
                        pred, ff = self.finalblocks[i+j](aux_emb)
                        ffs.append(ff)
                        preds.append(pred)
                    else:
                        ff = self.finalblocks[i+j](aux_emb)
                        embs.append(aux_emb)
                        ffs.append(ff)
                else:
                    aux_emb = emb[:,int(2048/self.n_groups*j):int(2048/self.n_groups*(j+1))]
                    pred, ff = self.finalblocks[i+j](aux_emb)
                    embs.append(aux_emb)
                    ffs.append(ff)
                    preds.append(pred)
                    
        return preds, embs, ffs

    
class MBR_model(nn.Module):         
    def __init__(self, class_num, n_branches, n_groups, losses="LBS", backbone="ibn", droprate=0, linear_num=False, return_f = True, circle_softmax=False, pretrain_ongroups=True, end_bot_g=False, group_conv_mhsa=False, group_conv_mhsa_2=False, x2g=False, x4g=False, LAI=False, n_cams=0, n_views=0):
        super(MBR_model, self).__init__()  

        self.modelup2L3 = base_branches(backbone=backbone)
        self.modelL4 = multi_branches(n_branches=n_branches, n_groups=n_groups, pretrain_ongroups=pretrain_ongroups, end_bot_g=end_bot_g, group_conv_mhsa=group_conv_mhsa, group_conv_mhsa_2=group_conv_mhsa_2, x2g=x2g, x4g=x4g)
        self.finalblock = FinalLayer(class_num=class_num, n_branches=n_branches, n_groups=n_groups, losses=losses, droprate=droprate, linear_num=linear_num, return_f=return_f, circle_softmax=circle_softmax, LAI=LAI, n_cams=n_cams, n_views=n_views, x2g=x2g, x4g=x4g)
        

    def forward(self, x,cam, view):
        mix = self.modelup2L3(x)
        output = self.modelL4(mix)
        preds, embs, ffs = self.finalblock(output, cam, view)

        return preds, embs, ffs, output



if __name__ == "__main__":
    input = torch.randn((32,3,256,256))

    ### MBR_4B
    model = MBR_model(575, ["R50", "R50", "BoT", "BoT"], n_groups=0, losses ="LBS", LAI=True)
    preds, embs, ffs, output = model(input, torch.randint(0,19,(32,1)), torch.randint(0,7,(32,8)))
    print("\nn_preds: ", len(preds))
    print("n_embs: ", len(embs))
    print("ffs: ", len(ffs))

