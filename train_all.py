"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License

modified by Heejoon Lee (2020/11/30)
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import csv

import utils
from model.position_emb import prepare_graph_variables_all


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(
                                logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels, device):
    # argmax
    logits = torch.max(logits, 1)[1].data
    logits = logits.view(-1, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, args, device=torch.device("cuda")):
    N = len(train_loader.dataset)
    lr_default = args.base_lr
    num_epochs = args.epochs
    lr_decay_epochs = range(args.lr_decay_start, num_epochs,
                            args.lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,
                            1.5 * lr_default, 2.0 * lr_default]

    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=lr_default, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=args.weight_decay) 

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f,'
                 % (lr_default, args.lr_decay_step,
                    args.lr_decay_rate) + 'grad_clip=%.2f' % args.grad_clip)
    logger.write('LR decay epochs: '+','.join(
                                        [str(i) for i in lr_decay_epochs]))
    last_eval_score, eval_score = 0, 0
    relation_type = train_loader.dataset.relation_type

    for epoch in range(0, num_epochs):
        pbar = tqdm(total=len(train_loader))
        total_norm, count_norm = 0, 0
        total_loss, train_score = 0, 0
        count, average_loss, att_entropy = 0, 0, 0
        t = time.time()
        if epoch < len(gradual_warmup_steps):
            for i in range(len(optim.param_groups)):
                optim.param_groups[i]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' %
                         optim.param_groups[-1]['lr'])
        elif (epoch in lr_decay_epochs or
              eval_score < last_eval_score and args.lr_decay_based_on_val):
            for i in range(len(optim.param_groups)):
                optim.param_groups[i]['lr'] *= args.lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[-1]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[-1]['lr'])
        last_eval_score = eval_score

        mini_batch_count = 0
        batch_multiplier = args.grad_accu_steps
        for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix,
                sem_adj_matrix) in enumerate(train_loader):
            batch_size = v.size(0)
            num_objects = v.size(1)
            if mini_batch_count == 0:
                optim.step()
                optim.zero_grad()
                mini_batch_count = batch_multiplier

            v = Variable(v).to(device)
            norm_bb = Variable(norm_bb).to(device)
            q = Variable(q).to(device)
            target = Variable(target).to(device)
            pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables_all(
                relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
                args.sem_label_num, device)
            pred, att, joint_att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                              spa_adj_matrix, target)
            loss = instance_bce_with_logits(pred, target)

            loss /= batch_multiplier
            loss.backward()
            mini_batch_count -= 1
            total_norm += nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
            count_norm += 1
            batch_score = compute_score_with_logits(pred, target, device).sum()
            total_loss += loss.data.item() * batch_multiplier * v.size(0)
            train_score += batch_score
            pbar.update(1)

            if args.log_interval > 0:
                average_loss += loss.data.item() * batch_multiplier
                if model.module.fusion == "ban":
                    current_att_entropy = torch.sum(calc_entropy(att.data))
                    att_entropy += current_att_entropy / batch_size / att.size(1)
                count += 1
                if i % args.log_interval == 0:
                    att_entropy /= count
                    average_loss /= count
                    print("step {} / {} (epoch {}), ave_loss {:.3f},".format(
                            i, len(train_loader), epoch,
                            average_loss),
                          "att_entropy {:.3f}".format(att_entropy))
                    print(joint_att.shape, joint_att)
                    average_loss = 0
                    count = 0
                    att_entropy = 0

        total_loss /= N
        train_score = 100 * train_score / N
        if eval_loader is not None:
            eval_score, bound, entropy = evaluate(
                model, eval_loader, device, args)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f'
                     % (total_loss, total_norm / count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)'
                         % (100 * eval_score, 100 * bound))

            if entropy is not None:
                info = ''
                for i in range(entropy.size(0)):
                    info = info + ' %.2f' % entropy[i]
                logger.write('\tentropy: ' + info)
        if (eval_loader is not None)\
           or (eval_loader is None and epoch >= args.saving_epoch):
            logger.write("saving current model weights to folder")
            model_path = os.path.join(args.output, 'model_%d.pth' % epoch)
            opt = optim if args.save_optim else None
            utils.save_model(model_path, model, epoch, opt)


@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    score = 0
    upper_bound = 0
    num_data = 0
    N = len(dataloader.dataset)
    entropy = None
    if model.module.fusion == "ban":
        entropy = torch.Tensor(model.module.glimpse).zero_().to(device)
    pbar = tqdm(total=len(dataloader))

    for i, (v, norm_bb, q, target, q_id, img_id, bb, spa_adj_matrix,
            sem_adj_matrix) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        norm_bb = Variable(norm_bb).to(device)
        q = Variable(q).to(device)
        target = Variable(target).to(device)

        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables_all(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        pred, att, joint_att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix, target)
        scores = compute_score_with_logits(pred, target, device)
        batch_score = scores.sum()
        
        score += batch_score
        upper_bound += (target.max(1)[0]).sum()
        num_data += pred.size(0)
        
        # Prints joint weights at every log_interval
#        print("Joint weights shape:", joint_att.shape)
#        print("Questions shape:", q_id.shape)
#        print("Image ID shape:", img_id.shape)
#         if i % args.log_interval == 0:
#             print("Joint weights: ", joint_att.shape, joint_att)
#             print("Questions: ", q_id.shape, q_id)
#             print("Images: ", img_id.shape, img_id)
            
#             for batch_num in range(batch_size):
#                 print("Example_number: ", batch_num)
#                 print("pred_idx: ", torch.argmax(pred[batch_num]))
#                 for answer_idx in range(3129):
#                     if (target[batch_num][answer_idx] > 0):
#                         print("answer_idx: ", answer_idx, \
#                               "answer_score: ", target[batch_num][answer_idx])
                    
                        
        
        if att is not None and 0 < model.module.glimpse\
                and entropy is not None:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
        pbar.update(1)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return score, upper_bound, entropy


@torch.no_grad()
def evaluate_single(model, dataloader, device, args, label2ans):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    score = 0
    upper_bound = 0
    num_data = 0
    N = len(dataloader.dataset)
    entropy = None
    if model.module.fusion == "ban":
        entropy = torch.Tensor(model.module.glimpse).zero_().to(device)
    pbar = tqdm(total=len(dataloader))
    
    pred_list = []

    for i, (v, norm_bb, q, target, q_id, img_id, bb, spa_adj_matrix,
            sem_adj_matrix) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        norm_bb = Variable(norm_bb).to(device)
        q = Variable(q).to(device)
        target = Variable(target).to(device)
        

        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables_all(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        pred, att, _, _ = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix, target)
        scores = compute_score_with_logits(pred, target, device)
        batch_score = scores.sum()
        
        score += batch_score
        upper_bound += (target.max(1)[0]).sum()
        num_data += pred.size(0)
        
        pred_list.append(pred)
        #predictions = torch.argmax(pred, 1)
        #prediction_list = predictions.tolist()
        #prediction_string = [[label2ans[i]] for i in prediction_list]
        
        #file = open('./data_spa.csv', 'a+', newline = '')
        #with file:
        #    write = csv.writer(file)
        #    write.writerows(prediction_string)
        
        # Prints joint weights at every log_interval
#        print("Joint weights shape:", joint_att.shape)
#        print("Questions shape:", q_id.shape)
#        print("Image ID shape:", img_id.shape)
#         if i % args.log_interval == 0:
#             print("Joint weights: ", joint_att.shape, joint_att)
#             print("Questions: ", q_id.shape, q_id)
#             print("Images: ", img_id.shape, img_id)
            
#             for batch_num in range(batch_size):
#                 print("Example_number: ", batch_num)
#                 print("pred_idx: ", torch.argmax(pred[batch_num]))
#                 for answer_idx in range(3129):
#                     if (target[batch_num][answer_idx] > 0):
#                         print("answer_idx: ", answer_idx, \
#                               "answer_score: ", target[batch_num][answer_idx])
                    
                        
        
        if att is not None and 0 < model.module.glimpse\
                and entropy is not None:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
        pbar.update(1)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return score, upper_bound, entropy, pred_list


@torch.no_grad()
def evaluate_v(model, dataloader, device, args, label2ans, questions):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    score = 0
    upper_bound = 0
    num_data = 0
    N = len(dataloader.dataset)
    entropy = None
    if model.module.fusion == "ban":
        entropy = torch.Tensor(model.module.glimpse).zero_().to(device)
    pbar = tqdm(total=len(dataloader))

    for i, (v, norm_bb, q, target, q_id, img_id, bb, spa_adj_matrix,
            sem_adj_matrix) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        norm_bb = Variable(norm_bb).to(device)
        q = Variable(q).to(device)
        target = Variable(target).to(device)

        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables_all(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        pred, att, joint_att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix, target)
        scores = compute_score_with_logits(pred, target, device)
        batch_score = scores.sum()
        
        score += batch_score
        upper_bound += (target.max(1)[0]).sum()
        num_data += pred.size(0)
        
        #Prints information of every single example
#         print("Joint weights: ", joint_att.shape)
        
        # Reshape joint_att matrix [_,_,1] -> [3, batch_size, 1]
        n_device = joint_att.shape[0]/3
        if n_device == 1:
            re_joint = joint_att
        elif n_device == 2:
            b = torch.zeros_like(joint_att)
            b[0] = joint_att[0]
            b[1] = joint_att[3]
            b[2] = joint_att[1]
            
            b[3] = joint_att[4]
            b[4] = joint_att[2]
            b[5] = joint_att[5]
            
            re_joint = b.view(3, -1, 1)
        elif n_device == 3:
            b = torch.zeros_like(joint_att)
            b[0] = joint_att[0]
            b[1] = joint_att[3]
            b[2] = joint_att[6]
            
            b[3] = joint_att[1]
            b[4] = joint_att[4]
            b[5] = joint_att[7]
            
            b[6] = joint_att[2]
            b[7] = joint_att[5]
            b[8] = joint_att[8]
       
            re_joint = b.view(3, -1, 1)
        else:
            print("Too many devices!")
            re_joint = joint_att
        
#         print("Re_joint weights:", re_joint.shape)
#         print("Questions: ", q_id.shape)
#         print("Images: ", img_id.shape)
        
        
        for batch in range(batch_size):
            example_data = [[batch+i*batch_size]]
            correct = False
            prediction = torch.argmax(pred[batch]).item()
            question_id = q_id[batch].item()
            question = ''
            for q in range(len(questions)):
                if questions[q]['question_id'] == question_id:
                    question = questions[q]['question']
                    break
            prediction_string = label2ans[prediction]
#             print(batch+i*batch_size,   # EX, img_id, q_id, q, imp, sem, spa, pred_id, pred, ans
#                   img_id[batch].item(), 
#                   question_id,
#                   question,
#                   re_joint[0][batch][0].item(), 
#                   re_joint[1][batch][0].item(), 
#                   re_joint[2][batch][0].item(),
#                   prediction,
#                   prediction_string)
            example_data[0].append(img_id[batch].item())
            example_data[0].append(question_id)
            example_data[0].append(question)
            example_data[0].append(re_joint[0][batch][0].item())
            example_data[0].append(re_joint[1][batch][0].item())
            example_data[0].append(re_joint[2][batch][0].item())
            example_data[0].append(prediction)
            example_data[0].append(prediction_string)
            answer_list = []
            answer_scores = []
            for answer_idx in range(3129):
                     if (target[batch][answer_idx] > 0):
#                         print("answer_idx: ", label2ans[answer_idx],
#                                "answer_score: ", target[batch][answer_idx])
                        answer_list.append(label2ans[answer_idx])
                        answer_scores.append(target[batch][answer_idx].item())
                        if prediction == answer_idx:
                            correct = True
            if correct == True:
#                 print("CORRECT!")
                example_data[0].append(1)
            else:
#                 print("WRONG!")
                example_data[0].append(0)
            
            for answer in range(len(answer_list)):
                example_data[0].append(answer_list[answer])
                example_data[0].append(answer_scores[answer])
            
            # Write to a file
            file = open('./data.csv', 'a+', newline = '')
            with file:
                write = csv.writer(file)
                write.writerows(example_data)
                         
            
        if att is not None and 0 < model.module.glimpse\
                and entropy is not None:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
        pbar.update(1)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return score, upper_bound, entropy


def calc_entropy(att):
    # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + eps).log()).sum(2).sum(0)  # g
