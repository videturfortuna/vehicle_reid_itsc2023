from turtle import update
import torch
import torch.nn.functional as F
from models.models import MBR_model
from tqdm import tqdm
import numpy as np
from metrics.eval_reid import eval_func




def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def get_model(data, device):

    ### 2B hybrid No LBS   
    if 'Hybrid_2B' == data['model_arch']:
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "BoT"], n_groups=0, losses="Classical", LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # resnet50 = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=data['branch_layer'], linear_num=False, pool=data['pool'])
        # trans_b = botnet_branch_oficial(class_num=data['n_classes'], droprate=0.0, linear_num=False, circle=True, branch_layer=data['branch_layer'], classifier=True)        
        # model = reid_baseline_2B_trans(resnet50, trans_b).to(device)

    ### 2B R50 No LBS
    if 'R50_2B' == data['model_arch']:
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # resnet50 = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=3, linear_num=False, pool=data['pool'])
        # subnet = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'], linear_num=data['linear_num'], backbone=data['backbone'], classifier=True,  n_classes=data['n_classes'])
        # model = reid_baseline_2B(resnet50, subnet).to(device)

    ### 2B R50 LBS
    if data['model_arch'] == 'MBR_R50_2B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # resnet50 = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=data["branch_layer"], linear_num=data['linear_num'], pool=data['pool'], backbone=data['backbone'])
        # subnet = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=data["branch_layer"], pool=data['pool_tripletb'], linear_num=data['linear_num'], end_BoT=False, backbone=data['backbone'])
        # model = reid_doublebranch(resnet50, subnet).to(device)


    ### Baseline with BoT
    if data['model_arch'] == 'BoT_baseline':
        model = MBR_model(class_num=data['n_classes'], n_branches=["BoT"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=0, linear_num=False, end_BoT=True, pool=data['pool']).to(device)

    ### 2B BoT LBS
    if data['model_arch'] == 'MBR_BOT_2B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # resnet50 = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=0, linear_num=False, end_BoT=True, pool=data['pool'])
        # trans_b = botnet_branch_oficial(class_num=data['n_classes'], droprate=0.0, linear_num=False, circle=True, classifier=False) # , pool=data['pool_trans']
        # model = reid_doublebranch_loss_split_BoT(resnet50, trans_b).to(device)

    ### MBR-4B (4B hybrid LBS)
    if data['model_arch'] == 'MBR_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # resnet50 = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=3, linear_num=False, pool=data['pool'])
        # subnet = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'])
        # trans_b = botnet_branch_oficial(class_num=data['n_classes'], droprate=0.0, linear_num=False, circle=True, classifier=True) # , pool=data['pool_trans']
        # trans_b_t = botnet_branch_oficial(class_num=data['n_classes'], droprate=0.0, linear_num=False, circle=True, classifier=False) # , pool=data['pool_trans']
        # model = reid_4B(resnet50, subnet, trans_b, trans_b_t).to(device)
    
    ### 4B hybdrid No LBS
    if data['model_arch'] == 'Hybrid_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "BoT", "BoT"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # resnet50 = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=3, linear_num=False, pool=data['pool'])
        # subnet = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'], classifier=True, n_classes=data['n_classes'])
        # trans_b = botnet_branch_oficial(class_num=data['n_classes'], droprate=0.0, linear_num=False, circle=True, classifier=True) # , pool=data['pool_trans']
        # trans_b_t = botnet_branch_oficial(class_num=data['n_classes'], droprate=0.0, linear_num=False, circle=True, classifier=True) # , pool=data['pool_trans']
        # model = reid_4B_bothlosses(resnet50, subnet, trans_b, trans_b_t).to(device)    
    
    ### 4B R50 No LBS
    if data['model_arch'] == 'R50_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # resnet50 = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=3, linear_num=False, pool=data['pool'])
        # subnet = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'], classifier=True, n_classes=data['n_classes'])
        # trans_b = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'], classifier=True, n_classes=data['n_classes']) # , pool=data['pool_trans']
        # trans_b_t = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'], classifier=True, n_classes=data['n_classes'])
        # model = reid_4B_bothlosses(resnet50, subnet, trans_b, trans_b_t).to(device)     

    
    if data['model_arch'] == 'MBR_R50_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # resnet50 = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, circle=True, branch_layer=3, linear_num=False, pool=data['pool'])
        # subnet = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'], classifier=False)
        # trans_b = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'], classifier=True, n_classes=data['n_classes']) # , pool=data['pool_trans']
        # trans_b_t = ResNet_IBNa_blocks_4e5(stride=1, branch_layer=3, pool=data['pool_tripletb'], classifier=False)
        # model = reid_4B_split_losses(resnet50, subnet, trans_b, trans_b_t).to(device)       
    


    ### 4G hybryd with LBS     MBR-4G
    if data['model_arch'] =='MBR_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        #model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], group_conv_mhsa=True, n_groups=4).to(device) 
    ### 4G hybrid No LBS
    if data['model_arch'] =='Hybrid_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        #model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], group_conv_mhsa=True, n_groups=4, all2losses=True).to(device) 



    # if data['model_arch'] =='MBR_2x2G_B':    
    #     model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, end_BoT=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], x2_2bg=True).to(device)    
    if data['model_arch'] =='MBR_2x2G':    
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x2g=True, group_conv_mhsa_2=True) 
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, end_BoT=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], x2_2bg=False, diff_inputs=True).to(device)    
    if data['model_arch'] =='MBR_R50_2x2G':  
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x2g=True)  
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, end_BoT=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], x2_2bg=False, diff_inputs=True, one_arch=True).to(device)  
    # if data['model_arch'] =='R50_2x2G':    
    #     model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, end_BoT=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], x2_2bg=False, diff_inputs=True, one_arch=True, all2losses=True).to(device)  
    # if data['model_arch'] == 'Hybrid_2x2G':  
    #     model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, end_BoT=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], x2_2bg=False, diff_inputs=True, one_arch=True, group_conv_mhsa_2=True).to(device)    

    ### 2G BoT LBS
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], end_bot_g=True)
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, end_BoT=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], end_bot_g =True).to(device)
    ### 2G R50 LBS
    if data['model_arch'] =='MBR_R50_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm']).to(device) 

    ### 2G Hybrid No LBS
    if data['model_arch'] =='Hybrid_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], group_conv_mhsa_2=True)
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, end_BoT=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], group_conv_mhsa_2=True, all2losses=True).to(device)

    ### 2G R50 No LBS
    if data['model_arch'] =='R50_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], all2losses=True).to(device)
    ### 4G R50 No LBS
    if data['model_arch'] =='R50_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], n_groups=4, all2losses=True).to(device)

    ### 4G only R50 with LBS
    if data['model_arch'] =='MBR_R50_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], group_conv_mhsa_2=True)
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], n_groups=4).to(device)    
    
    if data['model_arch'] =='MBR_R50_2x4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=["2x"], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x4g=True)
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], n_groups=4, x4g=True).to(device)       
    ### 2x4G with LBS
    if data['model_arch'] =='MBR_2x4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=["2x"], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x4g=True, group_conv_mhsa=True)
        # model = base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], groups=True, pretrain_ongroups=data['pretrain_ongroups'], groupnorm=data['group_norm'], n_groups=4, group_conv_mhsa=True, x4g=True).to(device)    
    

    if data['model_arch'] == 'Baseline':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        # model =  base_branches(class_num=data['n_classes'], droprate=0.0, stride=1, branch_layer=0, circle=True, linear_num=False, pool=data['pool'], backbone=data['backbone'], circle_softmax=data['circle_softmax']).to(device)  


    return model.to(device)



def train_epoch(model, device, dataloader, loss_fn, triplet_loss, optimizer, data, alpha_ce, beta_tri, logger, epoch, scheduler=None, scaler=False):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []
    ce_loss_log = []
    triplet_loss_log = []

    gamma_ce = data['gamma_ce']
    gamma_t = data['gamma_t']
    model_arch = data['model_arch']

    loss_log = tqdm(total=0, position=1, bar_format='{desc}', leave=True)
    loss_ce_log = tqdm(total=0, position=2, bar_format='{desc}', leave=True)
    loss_triplet_log = tqdm(total=0, position=3, bar_format='{desc}', leave=True)

    n_images = 0
    acc_v = 0
    stepcount = 0
    for image_batch, label, cam, view in tqdm(dataloader, desc='Epoch ' + str(epoch+1) +' (%)' , bar_format='{l_bar}{bar:20}{r_bar}'): 
        # Move tensor to the proper device
        loss_ce = 0
        loss_t = 0
        optimizer.zero_grad()

        image_batch = image_batch.to(device)
        label = label.to(device)
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):

                preds, embs, _, _ = model(image_batch, cam, view)
                loss = 0
                #### Losses 
                if type(preds) != list:
                    preds = [preds]
                    embs = [embs]
                for i, item in enumerate(preds):
                    if i%2==0 or "aseline" in model_arch or "R50" in model_arch:
                        loss_ce += alpha_ce * loss_fn(item, label)
                    else:
                        loss_ce += gamma_ce * loss_fn(item, label)
                for i, item in enumerate(embs):
                    if i%2==0 or "aseline" in model_arch or "R50" in model_arch:
                        loss_t += beta_tri * triplet_loss(item, label)
                    else:
                        loss_t += gamma_t * triplet_loss(item, label)

                if data['mean_losses']:
                    loss = loss_ce/len(preds) + loss_t/len(embs)
                else:
                    loss = loss_ce + loss_t
        else:
            preds, embs, ffs, activations = model(image_batch, cam, view)

            loss = 0
            #### Losses 
            if type(preds) != list:
                preds = [preds]
                embs = [embs]
            for i, item in enumerate(preds):
                if i%2==0 or "aseline" in model_arch or "R50" in model_arch:
                    loss_ce += alpha_ce * loss_fn(item, label)
                else:
                    loss_ce += gamma_ce * loss_fn(item, label)
            for i, item in enumerate(embs):
                if i%2==0 or "aseline" in model_arch or "R50" in model_arch:
                    loss_t += beta_tri * triplet_loss(item, label)
                else:
                    loss_t += gamma_t * triplet_loss(item, label)

            if data['mean_losses']:
                loss = loss_ce/len(preds) + loss_t/len(embs)
            else:
                loss = loss_ce + loss_t

        ###Training Acurracy
        for prediction in preds:
            acc_v += torch.sum(torch.argmax(prediction, dim=1) == label)
            n_images += prediction.size(0)
        stepcount += 1
    
        ### backward prop and optimizer step
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
      

        # train_acc.set_description_str(f'train loss (single batch): {acc_v/n_images:.3f}')
        loss_log.set_description_str(f'train loss : {loss.data:.3f}')
        loss_ce_log.set_description_str(f'CrossEntropy: {loss_ce.data:.3f}')
        loss_triplet_log.set_description_str(f'Triplet : {loss_t.data:.3f}')


        train_loss.append(loss.detach().cpu().numpy())
        ce_loss_log.append(loss_ce.detach().cpu().numpy())
        triplet_loss_log.append(loss_t.detach().cpu().numpy())


        logger.write_scalars({  "Loss/train_total": np.mean(train_loss), 
                                "Loss/train_crossentropy": np.mean(ce_loss_log),
                                "Loss/train_triplet": np.mean(triplet_loss_log),
                                "Loss/ce_loss_weight": alpha_ce,
                                "Loss/triplet_loss_weight": beta_tri,
                                "lr/learning_rate": get_lr(optimizer),
                                "Loss/AccuracyTrain": (acc_v/n_images).cpu().numpy()},
                                epoch * len(dataloader) + stepcount,
                                write_epoch=True
                                )


    print('\nTrain ACC (%): ', acc_v / n_images, "\n")
    
    return np.mean(train_loss), np.mean(ce_loss_log), np.mean(triplet_loss_log), alpha_ce, beta_tri



def test_epoch(model, device, dataloader_q, dataloader_g, model_arch, writer, epoch, remove_junk=True, scaler=False):
    model.eval()
    ###needed lists
    qf = []
    gf = []
    q_camids = []
    g_camids = []
    q_vids = []
    g_vids = []
    q_images = []
    g_images =  []

    with torch.no_grad():
        for image, q_id, cam_id, view_id in tqdm(dataloader_q, desc='Query infer (%)', bar_format='{l_bar}{bar:20}{r_bar}'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                _, _, ffs, _ = model(image, cam_id, view_id)
          
            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            qf.append(torch.cat(end_vec, 1))

            q_vids.append(q_id)
            q_camids.append(cam_id)

            if epoch == 119:  
                q_images.append(F.interpolate(image, (64,64)).cpu())
        #### TensorBoard emmbeddings
        if epoch == 119:    
            writer.write_embeddings(torch.cat(qf).cpu(), torch.cat(q_vids).cpu(), torch.cat(q_images)/2 + 0.5, 120, tag='Query embeddings')

        del q_images

        for image, q_id, cam_id, view_id in tqdm(dataloader_g, desc='Gallery infer (%)', bar_format='{l_bar}{bar:20}{r_bar}'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                    _, _, ffs, _ = model(image, cam_id, view_id)

            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            gf.append(torch.cat(end_vec, 1))
            g_vids.append(q_id)
            g_camids.append(cam_id)
        del g_images

    qf = torch.cat(qf, dim=0)
    gf = torch.cat(gf, dim=0)
    m, n = qf.shape[0], gf.shape[0]   
    distmat =  torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(),beta=1, alpha=-2)
    distmat = torch.sqrt(distmat).cpu().numpy()

    q_camids = torch.cat(q_camids, dim=0).cpu().numpy()
    g_camids = torch.cat(g_camids, dim=0).cpu().numpy()
    q_vids = torch.cat(q_vids, dim=0).cpu().numpy()
    g_vids = torch.cat(g_vids, dim=0).cpu().numpy()   
    
    del qf, gf
    
    cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)

    writer.write_scalars({"Accuraccy/CMC1": cmc[0], "Accuraccy/mAP": mAP}, epoch)

    return cmc, mAP
