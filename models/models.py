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




class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)



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



#### create conv4-5 Resnet Block with attention as BoTnet
class botnet_branch_oficial(nn.Module): 
    def __init__(self, class_num=770, droprate=0.0, linear_num=False, circle=True, use_mlp = False, classifier = True, GeM=False, batnorm=True, branch_layer=3):
        super(botnet_branch_oficial, self).__init__()

        if branch_layer == 3:
            layer_0 = Bottleneck_Transformer(1024, 512, resolution=[16, 16], use_mlp = use_mlp)
            layer_1 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = use_mlp)
            layer_2 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = use_mlp)

            self.model = nn.Sequential(layer_0, layer_1, layer_2)
        else:
            layer_0 = Bottleneck_Transformer(512, 256, resolution=[32, 32], stride=2, use_mlp = use_mlp)
            layer_1 = Bottleneck_Transformer(1024, 256, resolution=[16, 16], use_mlp = use_mlp)
            layer_2 = Bottleneck_Transformer(1024, 256, resolution=[16, 16], use_mlp = use_mlp)
            layer_3 = Bottleneck_Transformer(1024, 256, resolution=[16, 16], use_mlp = use_mlp)
            layer_4 = Bottleneck_Transformer(1024, 256, resolution=[16, 16], use_mlp = use_mlp)
            layer_5 = Bottleneck_Transformer(1024, 256, resolution=[16, 16], use_mlp = use_mlp)  
                      
            layer_6 = Bottleneck_Transformer(1024, 512, resolution=[16, 16], use_mlp = use_mlp)
            layer_7 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = use_mlp)
            layer_8 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = use_mlp)

            self.model = nn.Sequential(layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8)            
        # self.model.apply(weights_init_kaiming) 
        
        if GeM:
            self.avg_pool = GeneralizedMeanPooling()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.cl = classifier
        if classifier:
            self.classifier = ClassBlock(2048, class_num, droprate, bnorm=batnorm, linear=linear_num, return_f = circle)
        else: 
            self.classifier = nn.BatchNorm1d(2048)
            self.classifier.bias.requires_grad_(False)  # one model was trained with bias
            self.classifier.apply(weights_init_kaiming) 
        

    def forward(self, x):
        #if self.training:
        x = self.model(x)
        emb = self.avg_pool(x).squeeze()
        if self.cl:
            pred, emb_n = self.classifier(emb)
            return  pred, emb_n, emb, x 
        else:
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            emb_n = self.classifier(emb)
            
            return None, emb_n, emb, x





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
    def __init__(self, 
        class_num=30671, droprate=0.0, stride=1, branch_layer=2, circle=True, fdim=None, 
        linear_num=False, pool = 'GAP', end_BoT=False, splits=4,  no_classifier=False, 
        backbone='ibn', groups=False, pretrain_ongroups=True, circle_softmax=False, groupnorm=False, 
        groups_share_L3 = False, end_bot_g=False, n_groups=2, x2_2bg=False, diff_inputs=False, one_arch=False, group_conv_mhsa=False, end_groups=False, end_group_share=False, all2losses=False, group_conv_mhsa_2=False):
        super(base_branches, self).__init__()
        if backbone == 'r50':
            model_ft = models.resnet50()
        elif backbone == '101ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        elif backbone == '34ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        else:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True) #ResNet101_Weights.IMAGENET1K_V2
        
        assert branch_layer in [0,2,3,4]
        

        self.groups = groups
        self.groups_share_L3 = groups_share_L3

        self.x2_2bg = x2_2bg
        self.diff_inputs = diff_inputs
        self.end_groups = end_groups
        self.end_group_share = end_group_share
        self.all2losses = all2losses
        if n_groups > 2:
            self.n_4groups = True
        else:
            self.n_4groups = False
        self.group_conv_mhsa = group_conv_mhsa
        self.group_conv_mhsa_2 = group_conv_mhsa_2

        if groups:
            convlist = [k.split('.') for k, m in model_ft.layer4.named_modules(remove_duplicate=False) if isinstance(m, nn.Conv2d)]
            for item in convlist:
                if item[1] == "downsample":
                    m = model_ft.layer4[int(item[0])].get_submodule(item[1])[0]
                else:
                    m = model_ft.layer4[int(item[0])].get_submodule(item[1]) #'.'.join(
                weight = m.weight[:int(m.weight.size(0)), :int(m.weight.size(1)/n_groups), :,:]
                if end_bot_g and item[1]=="conv2":
                    setattr(model_ft.layer4[int(item[0])], item[1], MHSA_2G(int(512), int(512)))
                elif group_conv_mhsa and item[1]=="conv2":
                    setattr(model_ft.layer4[int(item[0])], item[1], Conv_MHSA_4G(int(512), int(512)))
                elif group_conv_mhsa_2 and item[1]=="conv2":
                    setattr(model_ft.layer4[int(item[0])], item[1], Conv_MHSA_2G(int(512), int(512)))
                else:
                    if item[1] == "downsample":
                        getattr(model_ft.layer4[int(item[0])], item[1])[0] = nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming)
                        if pretrain_ongroups:
                            getattr(model_ft.layer4[int(item[0])], item[1])[0].weight.data = weight
                    elif item[1] == "conv2":
                        setattr(model_ft.layer4[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=3, stride=1, padding=(1,1), groups=n_groups, bias=False).apply(weights_init_kaiming))
                        if pretrain_ongroups:
                            setattr(model_ft.layer4[int(item[0])].get_submodule(item[1]).weight, "data", weight)                        
                    else:
                        setattr(model_ft.layer4[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming))
                        if pretrain_ongroups:
                            setattr(model_ft.layer4[int(item[0])].get_submodule(item[1]).weight, "data", weight)

        if x2_2bg or diff_inputs:
            if one_arch:
                self.replica_bot_l4 = copy.deepcopy(model_ft.layer4)
            else:
                self.replica_bot_l4 = copy.deepcopy(model_ft.layer4)
                self.replica_bot_l4[0].conv2 = MHSA_2G(int(512),int(512))
                self.replica_bot_l4[1].conv2 = MHSA_2G(int(512),int(512))
                self.replica_bot_l4[2].conv2 = MHSA_2G(int(512),int(512))

        # avg pooling to global pooling
        model_ft.layer4[0].downsample[0].stride = (stride,stride)
        model_ft.layer4[0].conv2.stride = (stride,stride)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            if backbone == "34ibn":
                model_ft.layer4[0].conv1.stride = (1,1)
            else:
                model_ft.layer4[0].conv2.stride = (1,1)
        if pool =='GeM':
            model_ft.avgpool = GeneralizedMeanPooling()
        elif pool =='GeMP':
            model_ft.avgpool = GeneralizedMeanPoolingP()
        else:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.pool = pool

        # -1 no fully connected, up to layer 3: -2, up to layer 2: -3
        model_ft = torch.nn.Sequential(*(list(model_ft.children())[:-1])) 


        if end_BoT:
            layer_0 = Bottleneck_Transformer(1024, 512, resolution=[16, 16], use_mlp = False)
            layer_1 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = False)
            layer_2 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = False)
            tmp = nn.Sequential(layer_0, layer_1, layer_2)
            model_ft[7] = tmp

        self.model = model_ft
        self.branch_layer = branch_layer


        if branch_layer == 4:
            self.classifier = ClassBlock(2048, class_num, droprate, final_dim=fdim, linear=linear_num, return_f = circle, circle=circle_softmax)
            self.classifier_t = nn.Linear(2048, 1024)
            self.classifier_t.apply(weights_init_kaiming)
            self.nnbatchnormvector= nn.BatchNorm1d(1024)
            self.nnbatchnormvector.bias.requires_grad_(False)  
            self.nnbatchnormvector.apply(weights_init_kaiming)
        else:
            if pool == 'HMP':
                self.classifier =  nn.ModuleList()
                for i in range(splits):
                    self.classifier.append(ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle))
            else:
                if no_classifier:
                    self.no_classifier = no_classifier
                else:
                    self.no_classifier = False
                    if groups:
                        if n_groups == 4:
                            if self.all2losses:
                                self.classifier_1 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.classifier_2 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.classifier_3 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.classifier_4 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax) 
                            else:
                                self.classifier_1 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.classifier_2 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)  
                                self.bn1= nn.BatchNorm1d(int(512))    
                                self.bn1.bias.requires_grad_(False)  
                                self.bn1.apply(weights_init_kaiming)
                                self.bn2= nn.BatchNorm1d(int(512))
                                self.bn2.bias.requires_grad_(False)  
                                self.bn2.apply(weights_init_kaiming)                                                                            
                        elif x2_2bg or diff_inputs:
                            if self.group_conv_mhsa_2 or self.all2losses:
                                self.classifier_1 = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.classifier_2 = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.classifier_3 = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.classifier_4 = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)   
                            else:
                                self.classifier_1 = ClassBlock(int(1024), class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.bn1= nn.BatchNorm1d(int(1024))
                                self.bn1.bias.requires_grad_(False)  
                                self.bn1.apply(weights_init_kaiming)
                                self.classifier_2 = ClassBlock(int(1024), class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                                self.bn2= nn.BatchNorm1d(int(1024))
                                self.bn2.bias.requires_grad_(False)  
                                self.bn2.apply(weights_init_kaiming)
                        elif self.all2losses:
                            self.classifier_1 = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                            self.classifier_2 = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                        else:
                            self.bn1= nn.BatchNorm1d(1024)
                            self.bn1.bias.requires_grad_(False)  
                            self.bn1.apply(weights_init_kaiming)
                            self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)
                    else:
                        if backbone=="34ibn":
                            self.classifier = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax) 
                        else:
                            self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle, circle=circle_softmax)  

    def forward(self, x):
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)
        x = self.model[4](x)
        if self.branch_layer == 2:
            mix = self.model[5](x)
            x = self.model[6](mix)
            activations = self.model[7](x)
        elif self.branch_layer == 3:
            x = self.model[5](x)
            mix = self.model[6](x)
            activations = self.model[7](mix)
        else:
            x = self.model[5](x)
            mix = self.model[6](x)
            if self.diff_inputs:
                half = int(mix.size(1)/2)
                half_mix_2 = mix[:,half:,:,:]
                half_mix_2 = torch.cat((half_mix_2, half_mix_2), 1)
                activations = self.model[7](half_mix_2)
            else:
                activations = self.model[7](mix)
        x = self.model[8](activations)

        emb = x.view(x.size(0), x.size(1))

        if self.no_classifier:
            return emb, activations, mix
        else:
            if self.groups:
                if self.n_4groups:
                    if self.all2losses:
                        half = int(emb.size(1)/2)
                        quarter = int(emb.size(1)/4)
                        emb_cet_1 = emb[:, :quarter]
                        emb_cet_2 = emb[:, quarter:half]
                        emb_cet_3 = emb[:, half:half+quarter]
                        emb_cet_4 = emb[:, half+quarter:]                        
                        x , f = self.classifier_1(emb_cet_1)
                        x_2 , f_2 = self.classifier_2(emb_cet_2)
                        x_3 , f_3 = self.classifier_3(emb_cet_3)
                        x_4 , f_4 = self.classifier_4(emb_cet_4)

                        return [[x, x_3, x_2, x_4], [emb_cet_1, emb_cet_3, emb_cet_2, emb_cet_4], [f, f_2, f_3, f_4], [activations[:,:quarter,:,:], activations[:,quarter:half,:,:], activations[:,half:half+quarter,:,:], activations[:,half+quarter:,:,:]]]
                    
                    else:
                        half = int(emb.size(1)/2)
                        quarter = int(emb.size(1)/4)
                        emb_cet_1 = emb[:, :quarter]
                        emb_cet_2 = emb[:, quarter:half]
                        emb_cet_3 = emb[:, half:half+quarter]
                        emb_cet_4 = emb[:, half+quarter:]  

                        x_1 , f_1 = self.classifier_1(emb_cet_1)
                        x_3 , f_3 = self.classifier_2(emb_cet_3)
                        ff_2 = self.bn1(emb_cet_2) 
                        ff_4 = self.bn2(emb_cet_4) 

                        return [[x_1,x_3], [emb_cet_2, emb_cet_4], [f_1, ff_2, f_3, ff_4], [activations[:,:quarter,:,:], activations[:,quarter:half,:,:], activations[:,half:half+quarter,:,:], ff_4, emb_cet_4, activations[:,half+quarter:,:,:]]]
                        
                elif self.x2_2bg:
                    half = int(emb.size(1)/2)
                    activ_bot = self.replica_bot_l4(mix)
                    emb_bot = self.model[8](activ_bot)
                    emb_bot = emb_bot.view(emb_bot.size(0), emb_bot.size(1))

                    emb_bot_ce = emb_bot[:, :half]
                    emb_bot_t = emb_bot[:, half:]
                    emb_r50_ce = emb[:, :half]
                    emb_r50_t = emb[:, half:]

                    f_r50_t = self.bn1(emb_r50_t)
                    f_bot_t = self.bn2(emb_bot_t)

                    x_1 , f_1 = self.classifier_1(emb_r50_ce)
                    x_2 , f_2 = self.classifier_2(emb_bot_ce)

                    return [[x_1, x_2], [emb_r50_t, emb_bot_t], [f_1, f_r50_t, f_2, f_bot_t], [activations[:,:half,:,:], activations[:,half:,:,:], activ_bot[:,:half,:,:], activ_bot[:,half:,:,:]]]
                    
                elif self.diff_inputs:
                    if self.group_conv_mhsa_2 or self.all2losses:
                        half = int(mix.size(1)/2)
                        half_mix_1 = mix[:,:half,:,:]
                        half_mix_1 = torch.cat((half_mix_1, half_mix_1), 1)
                        activ_bot = self.replica_bot_l4(half_mix_1)
                        emb_bot = self.model[8](activ_bot)
                        emb_bot = emb_bot.view(emb_bot.size(0), emb_bot.size(1))

                        half_emb = int(emb.size(1)/2)
                        emb_cet_1 = emb_bot[:, :half_emb]  #r50
                        emb_cet_2 = emb_bot[:, half_emb:]  #bot
                        emb_cet_3 = emb[:, :half_emb]      #r50
                        emb_cet_4 = emb[:, half_emb:]      #bot

                        x , f = self.classifier_1(emb_cet_1)
                        x_2 , f_2 = self.classifier_2(emb_cet_2)
                        x_3 , f_3 = self.classifier_3(emb_cet_3)
                        x_4 , f_4 = self.classifier_4(emb_cet_4)

                        return [[x, x_2, x_3, x_4], [emb_cet_1, emb_cet_2, emb_cet_3, emb_cet_4], [f, f_2, f_3, f_4], [activations[:,:half_emb,:,:], activations[:,half_emb:,:,:], activ_bot[:,:half_emb,:,:], activ_bot[:,half_emb:,:,:]]]
                    else:
                        half = int(mix.size(1)/2)
                        half_mix_1 = mix[:,:half,:,:]
                        half_mix_1 = torch.cat((half_mix_1, half_mix_1), 1)
                        

                        activ_bot = self.replica_bot_l4(half_mix_1)
                        emb_bot = self.model[8](activ_bot)
                        emb_bot = emb_bot.view(emb_bot.size(0), emb_bot.size(1))

                        half_emb = int(emb.size(1)/2)

                        emb_bot_ce = emb_bot[:, :half_emb]
                        emb_bot_t = emb_bot[:, half_emb:]
                        emb_r50_ce = emb[:, :half_emb]
                        emb_r50_t = emb[:, half_emb:]

                        f_r50_t = self.bn1(emb_r50_t)
                        f_bot_t = self.bn2(emb_bot_t)

                        x_1 , f_1 = self.classifier_1(emb_r50_ce)
                        x_2 , f_2 = self.classifier_2(emb_bot_ce)

                        return[[x_1, x_2], [emb_r50_t, emb_bot_t], [f_1, f_2, f_r50_t, f_bot_t], [activations[:,:half_emb,:,:], activations[:,half_emb:,:,:], activ_bot[:,:half_emb,:,:], activ_bot[:,half_emb:,:,:]]]                       
                elif self.all2losses:
                    half = int(emb.size(1)/2)
                    emb_cet_1 = emb[:, :half]
                    emb_cet_2 = emb[:, half:]
                    x , f = self.classifier_1(emb_cet_1)
                    x_2 , f_2 = self.classifier_2(emb_cet_2)

                    return [[x, x_2], [emb_cet_1, emb_cet_2], [f, f_2], [activations[:,:half,:,:], activations[:,half:,:,:]]]                    
                else:
                    half = int(emb.size(1)/2)
                    emb_ce = emb[:, :half]
                    emb_t = emb[:, half:]
                    x , f = self.classifier(emb_ce)
                    fn_t = self.bn1(emb_t)
   
                    return [[x], [emb_t], [f, fn_t], [activations[:,:half,:,:], activations[:,half:,:,:]]]
            else:
                x , f = self.classifier(emb)
        if self.branch_layer == 4:
            ft = self.classifier_t(emb)
            ft_n = self.nnbatchnormvector(ft)

            return x, f, emb, activations, mix, ft, ft_n

        return x, f, emb, activations, mix


class ResNet_IBNa_blocks_4e5(nn.Module):
    def __init__(self, stride=1, branch_layer = 3, pool = 'GAP', linear_num=False, end_BoT=False, backbone='ibn', classifier=False, n_classes=0):
        super(ResNet_IBNa_blocks_4e5, self).__init__()
        #model_ft = models.resnet50(pretrained=True)
        assert branch_layer in [2,3]
        if backbone == 'ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True) # 'resnet50_ibn_a'
        else:
            model_ft = models.resnet50(weights="IMAGENET1K_V2")
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        if pool=="GeM":
            model_ft.avgpool = GeneralizedMeanPooling()
        elif pool =='GeMP':
            model_ft.avgpool = GeneralizedMeanPoolingP()
        else:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if branch_layer == 2:
            model_ft = torch.nn.Sequential(*(list(model_ft.children())[6:-1])) 
        if branch_layer == 3:
            if linear_num:
                model_ft = torch.nn.Sequential(*(list(model_ft.children())[7:-1]), nn.ModuleList([nn.Linear(2048, 2048).apply(weights_init_kaiming), nn.ReLU(), nn.Linear(2048, linear_num).apply(weights_init_kaiming)]))
            else:
                model_ft = torch.nn.Sequential(*(list(model_ft.children())[7:-1]))    
        if end_BoT:
            model_ft[0][0].conv2 = MHSA(512, width=16, height=16, heads=4).apply(weights_init_kaiming)
            model_ft[0][0].bn2 = nn.BatchNorm2d(512).apply(weights_init_kaiming)
            model_ft[0][1].conv2 = MHSA(512, width=16, height=16, heads=4).apply(weights_init_kaiming)
            model_ft[0][1].bn2 = nn.BatchNorm2d(512).apply(weights_init_kaiming)
            model_ft[0][2].conv2 = MHSA(512, width=16, height=16, heads=4).apply(weights_init_kaiming)
            model_ft[0][2].bn2 = nn.BatchNorm2d(512).apply(weights_init_kaiming)             
        self.model = model_ft
        self.classes = classifier

        
        if classifier:
            self.classifier = ClassBlock(2048, n_classes, 0, linear=linear_num, return_f = True)
        else:
            self.nnbatchnormvector= nn.BatchNorm1d(2048)
            self.nnbatchnormvector.bias.requires_grad_(False)  # no shift
            self.nnbatchnormvector.apply(weights_init_kaiming)

        self.branch_layer = branch_layer

     

        
    def _forward_impl(self, x: Tensor) -> Tensor:

        if self.branch_layer == 2:
            x = self.model[0](x)
            activations = self.model[1](x)
        if self.branch_layer == 3:
            activations = self.model[0](x)
        if self.branch_layer == 2:
            x = self.model[2](activations)
        if self.branch_layer == 3:
            x = self.model[1](activations)
        emb = torch.flatten(x, 1)
        if self.classes:
            pred, f = self.classifier(emb)
            return pred, f, emb, activations

        f = self.nnbatchnormvector(emb)
        return f, emb, activations
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)





class reid_doublebranch_loss_split_BoT(nn.Module):

    def __init__(self, ft_net, subnet):
        super(reid_doublebranch_loss_split_BoT, self).__init__()

        self.globalBranch = ft_net
        self.SubNet4e5 = subnet
        

    def forward(self, x):

        pred, f, emb, activations, mix = self.globalBranch(x)

        none ,f_s, emb_s, activations_triplet = self.SubNet4e5(mix)

        return [[pred], [emb_s], [f,f_s], [activations, activations_triplet]]




class reid_doublebranch(nn.Module):

    def __init__(self, ft_net, subnet):
        super(reid_doublebranch, self).__init__()

        self.globalBranch = ft_net
        self.SubNet4e5 = subnet
        

    def forward(self, x):

        pred, f, emb, activations, mix = self.globalBranch(x)

        f_s, emb_s, activations_triplet = self.SubNet4e5(mix)

        return [[pred], [emb_s], [f, f_s], [activations, activations_triplet]]



class reid_baseline_2B_trans(nn.Module):
    def __init__(self, res50, transformerB):
        super(reid_baseline_2B_trans, self).__init__()

        self.globalBranch = res50
        self.transform_net = transformerB

    def forward(self, x):

        pred, f, emb, activations, mix = self.globalBranch(x)
        pred_t, emb_n_t, emb_t, trans_activ = self.transform_net(mix)

        return [[pred, pred_t], [emb, emb_t], [f, emb_n_t], [activations, trans_activ]]







class reid_baseline_2B(nn.Module):
    def __init__(self, res50, transformerB):
        super(reid_baseline_2B, self).__init__()

        self.globalBranch = res50
        self.SubNet4e5 = transformerB

    def forward(self, x):

        pred, f, emb, activations, mix = self.globalBranch(x)
        pred_s, f_s, emb_s, activations_subnet = self.SubNet4e5(mix)

        return [[pred, pred_s], [emb, emb_s], [f, f_s], [activations, activations_subnet]]       



class reid_4B_bothlosses(nn.Module):
    def __init__(self, ft_net, subnet, t_branch, t_t_branch):
        super(reid_4B_bothlosses, self).__init__()

        self.globalBranch = ft_net
        self.SubNet4e5 = subnet
        self.transform_net = t_branch
        self.transform_net_t = t_t_branch


    def forward(self, x):
        pred_1, f_1, emb_1, activations_1, mix = self.globalBranch(x)

        pred_2, f_2, emb_2, activations_2 = self.SubNet4e5(mix)

        pred_3, f_3, emb_3, activations_3 = self.transform_net(mix)

        pred_4, f_4, emb_4, activations_4= self.transform_net_t(mix)

        return [[pred_1, pred_3, pred_2, pred_4], [emb_1, emb_3, emb_2, emb_4], [f_1, f_2, f_3, f_4],[activations_1, activations_2, activations_3, activations_4]]

    
class reid_4B_split_losses(nn.Module):
    def __init__(self, ft_net, subnet, t_branch, t_t_branch):
        super(reid_4B_split_losses, self).__init__()

        self.globalBranch = ft_net
        self.SubNet4e5 = subnet
        self.transform_net = t_branch
        self.transform_net_t = t_t_branch


    def forward(self, x):
        pred_1, f_1, emb_1, activations_1, mix = self.globalBranch(x)

        f_2, emb_2, activations_2 = self.SubNet4e5(mix)

        pred_3, f_3, emb_3, activations_3 = self.transform_net(mix)

        f_4, emb_4, activations_4 = self.transform_net_t(mix)

        return [[pred_1, pred_3], [emb_2, emb_4], [f_1, f_2, f_3, f_4], [activations_1, activations_2, activations_3, activations_4]]




class reid_4B(nn.Module):
    def __init__(self, ft_net, subnet, t_branch, t_t_branch):
        super(reid_4B, self).__init__()

        self.globalBranch = ft_net
        self.SubNet4e5 = subnet
        self.transform_net = t_branch
        self.transform_net_t = t_t_branch


    def forward(self, x):
        pred, f, emb, activations, mix = self.globalBranch(x)

        f_s, emb_s, activations_triplet = self.SubNet4e5(mix)

        pred_t, emb_n_t, emb_t, trans_activ_ce = self.transform_net(mix)

        pred_t_t, fn_t_bot, emb_t_bot, trans_activ_t = self.transform_net_t(mix)

        return [[pred, pred_t], [emb_s, emb_t_bot], [f, f_s, emb_n_t, fn_t_bot], [activations, activations_triplet, trans_activ_ce, trans_activ_t]]

    

# class reid_4B_LAI(nn.Module):
#     def __init__(self, ft_net, subnet, t_branch, bot_t_branch, n_cams=20, n_views=0):
#         super(reid_4B_LAI, self).__init__()

#         self.globalBranch = ft_net
#         self.SubNet4e5 = subnet
#         self.transform_net = t_branch
#         self.transform_net_t = bot_t_branch
#         if n_views > 0 and n_cams > 0:
#             self.camview = 'camview'
#         elif n_views > 0:
#             self.camview = 'view'
#         else:
#             self.camview = 'cam'

#         if self.camview == 'camview':
#             self.sie_embed_ce = nn.Parameter(torch.zeros(n_cams * n_views, 2048))
#             self.sie_embed_t = nn.Parameter(torch.zeros(n_cams * n_views, 2048))
#             self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_cams * n_views, 2048))
#             self.sie_embed_bot_t = nn.Parameter(torch.zeros(n_cams * n_views, 2048))
#             self.n_views = n_views
#         elif self.camview == 'view':
#             self.sie_embed_ce = nn.Parameter(torch.zeros(n_views, 2048))
#             self.sie_embed_t = nn.Parameter(torch.zeros(n_views, 2048))
#             self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_views, 2048))  
#             self.sie_embed_bot_t = nn.Parameter(torch.zeros(n_views, 2048))
#         else:
#             self.sie_embed_ce = nn.Parameter(torch.zeros(n_cams, 2048))
#             self.sie_embed_t = nn.Parameter(torch.zeros(n_cams, 2048))
#             self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_cams, 2048))
#             self.sie_embed_bot_t = nn.Parameter(torch.zeros(n_cams, 2048))

#     def forward(self, x, cams, view):

#         if self.camview == 'camview':
#             sie_ce = self.sie_embed_ce[cams * self.n_views + view]
#             sie_t = self.sie_embed_t[cams * self.n_views + view]
#             sie_t_ce = self.sie_embed_t_ce[cams * self.n_views + view]
#             sie_bot_t = self.sie_embed_bot_t[cams * self.n_views + view]
#         elif self.camview == 'view':   
#             sie_ce = self.sie_embed_ce[view]
#             sie_t = self.sie_embed_t[view]
#             sie_t_ce = self.sie_embed_t_ce[view]
#             sie_bot_t = self.sie_embed_bot_t[view]
#         else:    
#             sie_ce = self.sie_embed_ce[cams]
#             sie_t = self.sie_embed_t[cams]
#             sie_t_ce = self.sie_embed_t_ce[cams]
#             sie_bot_t = self.sie_embed_bot_t[cams]

#         pred, f, emb, activations, mix = self.globalBranch(x, sie_ce)

#         f_s, emb_s, activations_triplet = self.SubNet4e5(mix, sie_t)

#         pred_t, emb_n_t, emb_t, trans_activ_ce = self.transform_net(mix, sie_t_ce)

#         pred_t_t, fn_t_bot, emb_t_bot, trans_activ_t = self.transform_net_t(mix, sie_bot_t)

#         return pred, f, emb, f_s, emb_s, activations, activations_triplet, pred_t, emb_n_t, emb_t, fn_t_bot, emb_t_bot, trans_activ_ce, trans_activ_t



class reid_LAI_4BG(nn.Module):
    def __init__(self, ft_net, n_cams=20, n_views=0):
        super(reid_LAI_4BG, self).__init__()

        self.globalBranch = ft_net

        if n_views > 0 and n_cams > 0:
            self.camview = 'camview'
        elif n_views > 0:
            self.camview = 'view'
        else:
            self.camview = 'cam'

        if self.camview == 'camview':
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_cams * n_views, 512))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_cams * n_views, 512))
            self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_cams * n_views, 512))
            self.sie_embed_bot_t = nn.Parameter(torch.zeros(n_cams * n_views, 512))
            self.n_views = n_views
        elif self.camview == 'view':
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_views, 512))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_views, 512))
            self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_views, 512))  
            self.sie_embed_bot_t = nn.Parameter(torch.zeros(n_views, 512))
        else:
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_cams, 512))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_cams, 512))
            self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_cams, 512))
            self.sie_embed_bot_t = nn.Parameter(torch.zeros(n_cams, 512))

    def forward(self, x, cams, view):

        if self.camview == 'camview':
            sie_ce = self.sie_embed_ce[cams * self.n_views + view]
            sie_t = self.sie_embed_t[cams * self.n_views + view]
            sie_t_ce = self.sie_embed_t_ce[cams * self.n_views + view]
            sie_bot_t = self.sie_embed_bot_t[cams * self.n_views + view]
        elif self.camview == 'view':   
            sie_ce = self.sie_embed_ce[view]
            sie_t = self.sie_embed_t[view]
            sie_t_ce = self.sie_embed_t_ce[view]
            sie_bot_t = self.sie_embed_bot_t[view]
        else:    
            sie_ce = self.sie_embed_ce[cams]
            sie_t = self.sie_embed_t[cams]
            sie_t_ce = self.sie_embed_t_ce[cams]
            sie_bot_t = self.sie_embed_bot_t[cams]

        side = torch.cat((sie_ce, sie_t, sie_t_ce, sie_bot_t),dim=1)
        pred, f, emb, f_s, emb_s, activations, activations_triplet, pred_t, emb_n_t, emb_t, fn_t_bot, emb_t_bot, trans_activ_ce, trans_activ_t = self.globalBranch(x, side)

        return pred, f, emb, f_s, emb_s, activations, activations_triplet, pred_t, emb_n_t, emb_t, fn_t_bot, emb_t_bot, trans_activ_ce, trans_activ_t



#### create conv4-5 Resnet Block with attention as BoTnet
class botnet_branch_oficial_side(nn.Module): 
    def __init__(self, class_num=770, droprate=0.0, linear_num=False, circle=True, use_mlp = False, classifier = True, GeM=False):
        super(botnet_branch_oficial_side, self).__init__()

        layer_0 = Bottleneck_Transformer(1024, 512, resolution=[16, 16], use_mlp = use_mlp)
        layer_1 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = use_mlp)
        layer_2 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = use_mlp)

        self.model = nn.Sequential(layer_0, layer_1, layer_2)

        if GeM:
            self.avg_pool = GeneralizedMeanPooling()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.cl = classifier
        if classifier:
            self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle)
        else: 
            self.classifier = nn.BatchNorm1d(2048)
            self.classifier.bias.requires_grad_(False)  # no shift
            self.classifier.apply(weights_init_kaiming) 
        

    def forward(self, x, side):
        #if self.training:
        x = self.model(x)
        emb = self.avg_pool(x).squeeze()
        emb = emb + side
        if self.cl:
            pred, emb_n = self.classifier(emb)
            return  pred, emb_n, emb, x 
        else:
            emb_n = self.classifier(emb)
            return None, emb_n, emb, x






class reid_MultiBranch_transformer_side(nn.Module):
    def __init__(self, ft_net, subnet, t_branch, n_cams=20, n_views=0):
        super(reid_MultiBranch_transformer_side, self).__init__()

        self.globalBranch = ft_net
        self.SubNet4e5 = subnet
        self.transform_net = t_branch
        if n_views > 0 and n_cams > 0:
            self.camview = 'camview'
        elif n_views > 0:
            self.camview = 'view'
        else:
            self.camview = 'cam'

        if self.camview == 'camview':
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_cams * n_views, 2048))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_cams * n_views, 2048))
            self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_cams * n_views, 2048))
            self.n_views = n_views
        elif self.camview == 'view':
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_views, 2048))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_views, 2048))
            self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_views, 2048))  
        else:
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_cams, 2048))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_cams, 2048))
            self.sie_embed_t_ce = nn.Parameter(torch.zeros(n_cams, 2048))

    def forward(self, x, cams, view):

        if self.camview == 'camview':
            sie_ce = self.sie_embed_ce[cams * self.n_views + view]
            sie_t = self.sie_embed_t[cams * self.n_views + view]
            sie_t_ce = self.sie_embed_t_ce[cams * self.n_views + view]
        elif self.camview == 'view':   
            sie_ce = self.sie_embed_ce[view]
            sie_t = self.sie_embed_t[view]
            sie_t_ce = self.sie_embed_t_ce[view]
        else:    
            sie_ce = self.sie_embed_ce[cams]
            sie_t = self.sie_embed_t[cams]
            sie_t_ce = self.sie_embed_t_ce[cams]

        pred, f, emb, activations, mix = self.globalBranch(x, sie_ce)

        f_s, emb_s, activations_triplet = self.SubNet4e5(mix, sie_t)

        pred_t, emb_n_t, emb_t, trans_activ = self.transform_net(mix, sie_t_ce)

        return pred, f, emb, activations, mix, f_s, emb_s, activations_triplet, pred_t, emb_n_t, emb_t, trans_activ
    


class reid_LAI_2BG(nn.Module):

    def __init__(self, ft_net, n_cams=20, n_views=0): ##V776 20 8
        super(reid_LAI_2BG, self).__init__()

        self.globalBranch = ft_net
        if n_views > 0 and n_cams > 0:
            self.camview = 'camview'
        elif n_views > 0:
            self.camview = 'view'
        else:
            self.camview = 'cam'

        if self.camview == 'camview':
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_cams * n_views, 1024))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_cams * n_views, 1024))
            self.n_views = n_views
        elif self.camview == 'view':
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_views, 1024))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_views, 1024))
        else:
            self.sie_embed_ce = nn.Parameter(torch.zeros(n_cams, 1024))
            self.sie_embed_t = nn.Parameter(torch.zeros(n_cams, 1024))
 

    def forward(self, x, cams, view):
        if self.camview == 'camview':
            sie_ce = self.sie_embed_ce[cams * self.n_views + view]
            sie_t = self.sie_embed_t[cams * self.n_views + view]
        elif self.camview == 'view':   
            sie_ce = self.sie_embed_ce[view]
            sie_t = self.sie_embed_t[view]
        else:    
            sie_ce = self.sie_embed_ce[cams]
            sie_t = self.sie_embed_t[cams]

        side = torch.cat((sie_ce, sie_t),dim=1)
        pred, f, emb, f_s, emb_s, activations, activations_triplet = self.globalBranch(x, side)


        return pred, f, emb, activations, f_s, emb_s, activations_triplet 
      