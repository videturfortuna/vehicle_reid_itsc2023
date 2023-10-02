import torch
import torch.nn as nn
import numpy as np

from math import log



def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y, split=0, half=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
      split: When the CUDA memory is not sufficient, we can split the dataset into different parts
             for the computing of distance.
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    if split == 0:
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        distmat = xx + yy
        distmat.addmm_(x, y.t(), beta=1, alpha=-2)

    else:
        distmat = x.new(m, n)
        start = 0
        x = x.cuda()

        while start < n:
            end = start + split if (start + split) < n else n
            num = end - start

            sub_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, num) + \
                    torch.pow(y[start:end].cuda(), 2).sum(dim=1, keepdim=True).expand(num, m).t()
            # sub_distmat.addmm_(1, -2, x, y[start:end].t())
            sub_distmat.addmm_(x, y[start:end].cuda().t(), beta=1, alpha=-2)
            distmat[:, start:end] = sub_distmat.cpu()
            start += num

    distmat = distmat.clamp(min=1e-12).sqrt()  # for numerical stability
    return distmat


class KD_loss(torch.nn.Module):
    def __init__(self, loss_name, out_dim, warmup_teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp=1.0, teacher_temp=0.1, kl_weight=0.5):
        super().__init__()

        self.loss_name = loss_name
        self.s_temp = student_temp
        self.t_temp = teacher_temp

        self.lamb = kl_weight
        
    def forward(self, s_emb, t_emb, pred_s, pred_t):
        if self.loss_name == 'SKD+RL2':
            relu = torch.nn.ReLU()
            loss_rl2 = torch.sqrt(torch.mean((t_emb-s_emb)**2))

            s_emb = F.normalize(relu(s_emb)) # relu
            t_emb = F.normalize(relu(t_emb))

            s_mat =  torch.matmul(s_emb, torch.t(s_emb))
            t_mat =  torch.matmul(t_emb, torch.t(t_emb))

            eig_values_s, eig_vector_s = torch.linalg.eig(s_mat)
            eig_values_t, eig_vector_t = torch.linalg.eig(t_mat)

            log_eig_values_s = torch.log(eig_values_s)
            eigval_diag_s = torch.diag(log_eig_values_s)
            s_mat_log = eig_vector_s @ eigval_diag_s @ eig_vector_s.t()       

            log_eig_values_t= torch.log(eig_values_t)
            eigval_diag_t = torch.diag(log_eig_values_t)
            t_mat_log = eig_vector_t @ eigval_diag_t @ eig_vector_t.t()           
     
            loss = (torch.norm(s_mat_log - t_mat_log, p='fro') ** 2 )  + loss_rl2         
        if self.loss_name == 'SKD':
            relu = torch.nn.ReLU()
            s_emb = F.normalize(relu(s_emb)) # relu
            t_emb = F.normalize(relu(t_emb))

            s_mat =  torch.matmul(s_emb, torch.t(s_emb))
            t_mat =  torch.matmul(t_emb, torch.t(t_emb))

            eig_values_s, eig_vector_s = torch.linalg.eig(s_mat)
            eig_values_t, eig_vector_t = torch.linalg.eig(t_mat)

            log_eig_values_s = torch.log(eig_values_s)
            eigval_diag_s = torch.diag(log_eig_values_s)
            s_mat_log = eig_vector_s @ eigval_diag_s @ eig_vector_s.t()       

            log_eig_values_t= torch.log(eig_values_t)
            eigval_diag_t = torch.diag(log_eig_values_t)
            t_mat_log = eig_vector_t @ eigval_diag_t @ eig_vector_t.t()           
     
            loss = torch.norm(s_mat_log - t_mat_log, p='fro') ** 2

        if self.loss_name == 'KL_CE':
            t_emb = F.softmax(t_emb / self.t_temp, dim=-1).detach()
            s_emb = s_emb / self.s_temp
            _loss_ce =  torch.mean(torch.sum(-t_emb * F.log_softmax(s_emb, dim=-1), dim=-1))
            
            pred_t = F.softmax(pred_t / self.t_temp, dim=-1).detach()
            pred_s = pred_s / self.s_temp
            _loss_kl = self.lamb * self.t_temp * self.s_temp * torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(pred_s, dim=-1), pred_t)

            loss = (1 - self.lamb) * _loss_ce + self.lamb * self.t_temp * self.s_temp * _loss_kl

        if self.loss_name == 'KL_RL2':
            pred_t = F.softmax(pred_t / self.t_temp, dim=-1).detach()
            pred_s = pred_s / self.s_temp
            _loss_kl = self.lamb * self.t_temp * self.s_temp * torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(pred_s, dim=-1), pred_t)
            loss = _loss_kl + torch.sqrt(torch.mean((t_emb-s_emb)**2))

        if self.loss_name == 'KL_RL2_embonly':
            pred_t = F.softmax(t_emb / self.t_temp, dim=-1).detach()
            pred_s = s_emb / self.s_temp
            _loss_kl = self.lamb * self.t_temp * self.s_temp * torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(pred_s, dim=-1), pred_t)
            loss = _loss_kl + torch.sqrt(torch.mean((t_emb-s_emb)**2))
        
        if self.loss_name == 'CE_RL2_embonly':
            pred_t = F.softmax(t_emb / self.t_temp, dim=-1).detach()
            pred_s = s_emb / self.s_temp
            _loss_ce =  torch.mean(torch.sum(-pred_t * F.log_softmax(pred_s, dim=-1), dim=-1))

            loss = _loss_ce + torch.sqrt(torch.mean((t_emb-s_emb)**2))

        if self.loss_name == 'KL':
            t_emb = F.softmax(t_emb / self.t_temp, dim=-1).detach()
            s_emb = s_emb / self.s_temp
            loss = torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(s_emb, dim=-1), t_emb)

        if self.loss_name=='CE':
            t_emb = F.softmax(t_emb / self.t_temp, dim=-1).detach()
            s_emb = s_emb / self.s_temp
            loss =  torch.mean(torch.sum(-t_emb * F.log_softmax(s_emb, dim=-1), dim=-1))
               
        if self.loss_name=='RL2': 
            loss = torch.sqrt(torch.mean((t_emb-s_emb)**2))

        if self.loss_name=='L2': 
            loss = torch.mean((t_emb-s_emb)**2)

        if self.loss_name=='L1': 
            loss = torch.mean(torch.abs(t_emb-s_emb))   

        return loss



def hard_example_mining(dist_mat, labels, mask=None, return_inds=False, mask_view=None, same_view=None):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      mask: pytorch Tensor, with shape [N, N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    if not mask_view is None:
        mask_view[mask_view == 2] = 2 #view2compare
        mask_view[mask_view == 5] = 2 #view2compare
        mask_view[mask_view == 0] = 0
        mask_view[mask_view == 3] = 0
        mask_view[mask_view == 6] = 0
        mask_view[mask_view == 1] = 1
        mask_view[mask_view == 7] = 1
        mask_view[mask_view == 4] = 1
        if same_view == 0:
            is_sameview = mask_view.expand(N, N).ne(mask_view.expand(N, N).t())
            is_pos = torch.logical_and(is_pos, is_sameview)
            is_neg = torch.logical_and(is_neg, is_sameview)
        else:
            is_sameview = mask_view.expand(N, N).eq(mask_view.expand(N, N).t())
            is_pos = torch.logical_and(is_pos, is_sameview)
            is_neg = torch.logical_and(is_neg, is_sameview)
        

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    if mask is None:
        mask = torch.ones_like(dist_mat)

    aux_mat = torch.zeros_like(dist_mat)
    aux_mat[mask==0] -= 10
    dist_mat = dist_mat + aux_mat


    dist_ap, relative_p_inds = torch.max(dist_mat * is_pos, 1, keepdim=True)
    ## original aproach
    # dist_ap, relative_p_inds = torch.max(
    #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)


    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    # dist_mat[dist_mat == 0] += 10000  # 处理非法值。归一化后的最大距离为2
    aux_mat = torch.zeros_like(dist_mat)
    aux_mat[mask==0] += 10000
    dist_mat = dist_mat + aux_mat

    ## original aproach
    # dist_an, relative_n_inds = torch.min(
    #     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(dist_mat * is_neg + is_pos * 1e8, 1, keepdim=True)
    # shape [N]



    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an




class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)#, reduction='sum'
        else:
            self.ranking_loss = nn.SoftMarginLoss()#reduction='sum'

    def __call__(self, global_feat, labels, mask=None, normalize_feature=False, mask_view=None, same_view=None):
        """
        :param global_feat:
        :param labels:
        :param mask:  [N, N] 可见性mask。不可见的mask将不会被选择。若全部不可见，则对结果*0
        :param normalize_feature:
        :return:
        """
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels, mask=mask, mask_view=mask_view, same_view=same_view)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss #, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, keep_dim=False):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.keep_dim = keep_dim

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = inputs.new_zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data, 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.keep_dim:
            loss = (- targets * log_probs).sum(1)
        else:
            loss = (- targets * log_probs).mean(0).sum()
        return loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, num_ids=6, views=8, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.num_ids = int(num_ids)
        self.views = views

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        features = torch.nn.functional.normalize(features, dim=1)
        features = features.view(self.num_ids, self.views, -1)
        labels = labels.view(self.num_ids, self.views)[:,0]

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        # pos_per_sample=mask.sum(1) #B
        # pos_per_sample[pos_per_sample<1e-6]=1.0
        # mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample 
        

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




class SupConLoss_2(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,num_ids=6, views=8, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.num_ids = int(num_ids)
        self.views = views

    def forward(self, features, labels, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # susu
        features = features.view(self.num_ids, self.views, -1)
        labels = labels.view(self.num_ids, self.views)[:,0]

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # import ipdb; ipdb.set_trace()
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
        
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362
        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='int32')[y]
      

class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss      


import torch.nn.functional as F

def binarize(T, nb_classes):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def pairwise_mahalanobis(X, means, log_vars):
    """
    Computes pairwise squared Mahalanobis distances between X (data points) and a set of distributions
    :param X: [N, F] where N is the batch size and F is the feature dimension
    :param means: [C, F] C is the number of classes
    :param log_vars: [C, F] C is the number of classes, we assume a diagonal covariance matrix
    :return: pairwise squared Mahalanobis distances... [N, C, F] matrix
    i.e., M_ij = (x_i-means_j)\top * inv_cov_j * (x_i - means_j)
    """
    sz_batch = X.size(0)
    nb_classes = means.size(0)

    new_X = torch.unsqueeze(X, dim=1)  # [N, 1, F]
    new_X = new_X.expand(-1, nb_classes, -1)  # [N, C, F]

    new_means = torch.unsqueeze(means, dim=0)  # [1, C, F]
    new_means = new_means.expand(sz_batch, -1, -1)  # [N, C, F]

    # pairwise distances
    diff = new_X - new_means

    # convert log_var to covariance
    covs = torch.unsqueeze(torch.exp(log_vars), dim=0)  # [1, C, F]

    # the squared Mahalanobis distances
    M = torch.div(diff.pow(2), covs).sum(dim=-1)  # [N, C]

    return M


# Class Distributions to Hypergraph
class CDs2Hg(nn.Module):
    def __init__(self, nb_classes, sz_embed, tau=32, alpha=0.9):
        super(CDs2Hg, self).__init__()
        # Parameters (means and covariance)
        self.means = nn.Parameter(torch.Tensor(nb_classes, sz_embed).cuda())
        self.log_vars = nn.Parameter(torch.Tensor(nb_classes, sz_embed).cuda())

        # Initialization
        nn.init.kaiming_normal_(self.means, mode='fan_out')
        nn.init.kaiming_normal_(self.log_vars, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.tau = tau
        self.alpha = alpha

    def forward(self, X, T):
        mu = self.means
        log_vars = self.log_vars
        log_vars = F.relu(log_vars)

        # L2 normalize
        X = F.normalize(X, p=2, dim=-1)
        mu = F.normalize(mu, p=2, dim=-1)

        # Labels of each distributions (NxC matrix)
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)

        # Compute pairwise mahalanobis distances (NxC matrix)
        distance = pairwise_mahalanobis(X, mu, log_vars)

        # Distribution loss
        mat = F.softmax(-1 * self.tau * distance, dim=1)
        loss = torch.sum(mat * P_one_hot, dim=1)
        non_zero = loss != 0
        loss = -torch.log(loss[non_zero])

        # Hypergraph construction
        class_within_batch = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        exp_term = torch.exp(-1 * self.alpha * distance[:, class_within_batch])
        H = P_one_hot[:, class_within_batch] + exp_term * (1 - P_one_hot[:, class_within_batch])

        return loss.mean() #, H

# Hypergraph Neural Networks (AAAI 2019)
class HGNN(nn.Module):
    def __init__(self, nb_classes, sz_embed, hidden):
        super(HGNN, self).__init__()

        self.theta1 = nn.Linear(sz_embed, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lrelu = nn.LeakyReLU(0.1)

        self.theta2 = nn.Linear(hidden, nb_classes)

    def compute_G(self, H):
        # the number of hyperedge
        n_edge = H.size(1)
        # the weight of the hyperedge
        we = torch.ones(n_edge).cuda()
        # the degree of the node
        Dv = (H * we).sum(dim=1)
        # the degree of the hyperedge
        De = H.sum(dim=0)

        We = torch.diag(we)
        inv_Dv_half = torch.diag(torch.pow(Dv, -0.5))
        inv_De = torch.diag(torch.pow(De, -1))
        H_T = torch.t(H)

        # propagation matrix
        G = torch.chain_matmul(inv_Dv_half, H, We, inv_De, H_T, inv_Dv_half)

        return G

    def forward(self, X, H):
        G = self.compute_G(H)

        # 1st layer
        X = G.matmul(self.theta1(X))
        X = self.bn1(X)
        X = self.lrelu(X)

        # 2nd layer
        out = G.matmul(self.theta2(X))

        return out




class AdaptiveWeightedRankListLoss(nn.Module):
    def __init__(self, alpha=1.8, m_var = 1.0, tn=10, tp=0):
        super(AdaptiveWeightedRankListLoss, self).__init__()

        self.alpha = alpha
        self.m = m_var
        self.softmin = torch.nn.Softmin(dim = None)
        self.activation = nn.ReLU()
        self.Tn = tn
        self.Tp = tp
        self.lamba = 0.5

    def forward(self, global_feat, labels):
        # Implementation of the original proposed Loss in "Ranked List Loss for Deep Metric Learning"
        N = global_feat.size(0)

        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).fill_diagonal_(False)
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        d_ij = euclidean_dist(global_feat, global_feat)

        lm_ = (1- is_pos.type(torch.uint8))*self.activation(self.alpha-d_ij) + is_pos.type(torch.uint8)*self.activation(d_ij-(self.alpha - self.m))

        #### Otain Non-trivial masks for Positive and Negatives
        #  where P_set = d_ij > (alpha-m) and N_set = d_ij < alpha
        p_mask = d_ij > (self.alpha - self.m)
        p_mask = torch.logical_and(is_pos.fill_diagonal_(False), p_mask).type(torch.uint8)
        n_mask = d_ij < self.alpha
        n_mask = torch.logical_and(is_neg, n_mask).type(torch.uint8)
        ####weights
        w_ij_n = torch.exp(self.Tn * (self.alpha - d_ij)) * n_mask
        w_ij_p = torch.exp(self.Tp * (d_ij - (self.alpha-self.m))) * p_mask

        w_n_sum = w_ij_n.sum().clamp(min=1e-12, max=None)
        w_p_sum = w_ij_p.sum().clamp(min=1e-12, max=None)

        L_n = (w_ij_n / w_n_sum * lm_).sum()
        if L_n > 1.0 : 
            print('finalmente')
        L_p = (w_ij_p / w_p_sum* lm_).sum()

        loss = (1-self.lamba)*L_p + self.lamba*L_n

        return loss #/N
        
def rank_loss(dist_mat, labels, margin,alpha,tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])
        
        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]
        
        ap_is_pos = torch.clamp(torch.add(dist_ap,margin-alpha),min=0.0)
        ap_pos_num = ap_is_pos.size(0) +1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num))

        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval*(-1*an_less_alpha+alpha))
        an_weight_sum = torch.sum(an_weight) +1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_weight_sum)
        
        total_loss = total_loss+loss_ap+loss_an
    total_loss = total_loss*1.0/N
    return total_loss

class RankedLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    
    def __init__(self, margin=0.4, alpha=1.2, tval=1): #margin=1.0, alpha=1.8, tval=1
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        
    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        total_loss = rank_loss(dist_mat,labels,self.margin,self.alpha,self.tval)
        
        return total_loss


class pairwise_circleloss(nn.Module):
    def __init__(self, m=0.25, gamma=128):
        super(pairwise_circleloss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, embedding, targets):

        embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)

        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        logit_n = self.gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=64):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def convert_label_to_similarity(self, normed_feature, label):
        # normed_feature = F.normalize(normed_feature)
        
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

    def forward(self, feat, label):
        sp, sn = self.convert_label_to_similarity(feat, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        #loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        loss = torch.log(1 + torch.exp(logit_n).sum(dim=0) + torch.exp(logit_p).sum(dim=0))

        #loss = self.soft_plus(torch.exp(logit_n).sum(dim=0) + torch.exp(logit_p).sum(dim=0))

        return loss








class Linear(nn.Module):
    def __init__(self, num_classes, scale=128, margin=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.s = scale
        self.m = margin

    def forward(self, logits, targets):
        return logits.mul_(self.s)

    def extra_repr(self):
        return f"num_classes={self.num_classes}, scale={self.s}, margin={self.m}"


class CosSoftmax(Linear):
    r"""Implement of large margin cosine distance:
    """

    def forward(self, logits, targets):
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], self.m)
        logits[index] -= m_hot
        logits.mul_(self.s)
        return logits


class ArcSoftmax(Linear):

    def forward(self, logits, targets):
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], self.m)
        logits.acos_()
        logits[index] += m_hot
        logits.cos_().mul_(self.s)
        return logits


class CircleSoftmax(nn.Module):
    def __init__(self, num_classes, scale=64, margin=0.25):  # running with 32 0.1
        super().__init__()
        self.num_classes = num_classes
        self.s = scale
        self.m = margin
        
    def forward(self, logits, targets):
        alpha_p = torch.clamp_min(-logits.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(logits.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        # When use model parallel, there are some targets not in class centers of local rank
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], 1)

        logits_p = alpha_p * (logits - delta_p)
        logits_n = alpha_n * (logits - delta_n)

        logits[index] = logits_p[index] * m_hot + logits_n[index] * (1 - m_hot)

        neg_index = torch.where(targets == -1)[0]
        logits[neg_index] = logits_n[neg_index]

        logits.mul_(self.s)
        # logits_p = (logits_p[index] * m_hot) * self.s
        # n_ids2mean_p = logits_p.count_nonzero()
        # logits_n = (logits_n[index] * (1 - m_hot)) * self.s
        # n_ids2mean_n = logits_n.count_nonzero()
        
        # loss = torch.log(1 + torch.exp(logits_n[torch.where(logits_p == 0)]).sum() + torch.exp(logits_p[torch.where(logits_p != 0)]).sum())
        # # loss = F.softplus(torch.logsumexp(logits_n, dim=0) + torch.logsumexp(logits_p, dim=0)).mean()

        return logits #logits


class CircleLoss_myimplementation(nn.Module):
    def __init__(self, m=0.25, gamma=64):
        super(CircleLoss_myimplementation, self).__init__()
        self.m = m
        self.s = gamma
        #self.soft_plus = nn.Softplus()
        self.Op = 1 + self.m
        self.On = -self.m
        self.delta_p = 1-self.m
        self.delta_n = self.m


    def forward(self, x, labels, emb, fc):
        ff = F.normalize(emb)
        for W in fc.parameters():
            W = F.normalize(W, dim=1)

        wf = fc(ff)
        
        ###AM-softmax
        # numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        # excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        # denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        # L = numerator - torch.log(denominator)
        # return -torch.mean(L)

        # CircleLoss

        wf_detach = wf.detach()
        alpha_p = torch.clamp_min(-torch.diagonal(wf_detach.transpose(0, 1)[labels]) + self.Op, min=0.)
        alpha_n = torch.clamp_min(torch.cat([torch.cat((wf_detach[i, :y], wf_detach[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0) - self.On, min=0.)

        sp = torch.exp(-self.s * alpha_p * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.delta_p))
        sn = torch.exp(self.s * alpha_n * (torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0) - self.delta_n))
        
        loss = torch.log(1 + sn.sum()*sp.sum())

        return loss










if __name__ == "__main__":
    bs = 32
    size_f = 2048
    relu = torch.nn.ReLU()
    batch_f = torch.randn((bs, size_f))
    labels = torch.tensor([1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4])
    labels_pred = relu(torch.randn((bs)))



    # loss_triplet = TripletLoss(margin=0.3)
    # loss_value = loss_triplet(batch_f, labels)

    loss_wrll = AdaptiveWeightedRankListLoss()
    loss_value = loss_wrll(batch_f, labels)




def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def gse_loss(dist_mat, dist_mat_st, dist_mat_at, labels, margin,alpha,tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    #dist_mat_st = torch.from_numpy(dist_mat_st_np)
    #dist_mat_at = torch.from_numpy(dist_mat_at_np)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])
        
        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]

        dist_ap_st = dist_mat_st[ind][is_pos]
        dist_an_at = dist_mat_at[ind][is_neg]
        
        ap_is_pos = torch.clamp(torch.add(dist_ap,margin-alpha),min=0.0)
        ap_st_weight = torch.exp(tval*(-1/(dist_ap_st+1e-5)))
        ap_pos_num = ap_is_pos.size(0) +1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos*ap_st_weight)
        loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num))

        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        weight_less_alpha = dist_an_at[an_is_pos]
        an_weight = torch.exp(tval*(-(weight_less_alpha +1e-5)))
        an_pos_num = an_is_pos.size(0) +1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_pos_num)

        
        total_loss = total_loss + loss_ap + loss_an
        #pdb.set_trace()
    total_loss = total_loss*1.0/N
    return total_loss

class Gse_Loss(object):
    "GSE Loss"
    
    def __init__(self, margin=0.7, alpha=1, tval=1.0):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        
    def __call__(self, global_feat, labels, cam_feat, view_feat, type_feat, color_feat, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            cam_feat    = normalize(cam_feat, axis=-1)
            view_feat   = normalize(view_feat, axis=-1)
            type_feat   = normalize(type_feat, axis=-1)
            color_feat  = normalize(color_feat, axis=-1)

        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_mat_st = (euclidean_dist(cam_feat, cam_feat) + euclidean_dist(view_feat, view_feat))/2
        dist_mat_at = (euclidean_dist(type_feat, type_feat) + euclidean_dist(color_feat, color_feat))/2

        total_loss = gse_loss(dist_mat, dist_mat_st, dist_mat_at, labels, self.margin, self.alpha, self.tval)
        
        return total_loss

class cross_entropy_loss(nn.Module):
    def __init__(self, eps, alpha=0.2):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.tau = 1 ##Temperature

    def forward(self, pred_class_outputs, gt_classes):
        num_classes = pred_class_outputs.size(1)

        if self.eps >= 0:
            smooth_param = self.eps
        else:
            # Adaptive label smooth regularization
            soft_label = F.softmax(pred_class_outputs, dim=1)
            smooth_param = self.alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

        log_probs = F.log_softmax(pred_class_outputs / self.tau, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).sum(dim=1)

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum() / non_zero_cnt

        return loss


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining_fastreid(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


class SPD(nn.Module):
    def __init__(self, hard_mining) -> None:
        super().__init__()
        self.hard_mining=hard_mining

    def forward(self, embedding, targets):
        dist_mat = torch.matmul(F.normalize(embedding), F.normalize(embedding).T)
        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        if self.hard_mining:
            dist_ap, _ = torch.min(dist_mat * is_pos + is_neg * 1e9, dim=1)
            dist_an, _ = torch.max(dist_mat * is_neg, dim=1)

        loss = (dist_an - dist_ap).mean()

        return loss


def euclidean_dist_fast_reid(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class triplet_loss_fastreid(nn.Module):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    def __init__(self, margin, norm_feat, hard_mining) -> None:
        super().__init__()
        self.margin = margin
        self.norm_feat=norm_feat
        self.hard_mining = hard_mining

    def forward(self, embedding, targets):
        if self.norm_feat:
            dist_mat = euclidean_dist_fast_reid(F.normalize(embedding), F.normalize(embedding))
            # dist_mat = torch.matmul(F.normalize(embedding), F.normalize(embedding).T)
        else:
            dist_mat = euclidean_dist_fast_reid(embedding, embedding)

        # For distributed training, gather all features from different process.
        # if comm.get_world_size() > 1:
        #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
        #     all_targets = concat_all_gather(targets)
        # else:
        #     all_embedding = embedding
        #     all_targets = targets

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        if self.hard_mining:
            dist_ap, dist_an = hard_example_mining_fastreid(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on

        return loss