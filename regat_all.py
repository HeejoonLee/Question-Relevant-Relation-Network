"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.

Modified by Heejoon Lee (2020/11/23)

All three relation encoders are trained simultaneously.
Attention layer is added to combine the three relation representations into one.

"""
import torch
import torch.nn as nn
from model.fusion import BAN, BUTD, MuTAN
from model.language_model import WordEmbedding, QuestionEmbedding,\
                                 QuestionSelfAttention
from model.relation_encoder import ImplicitRelationEncoder,\
                                   ExplicitRelationEncoder
from model.classifier import SimpleClassifier


class ReGAT_All(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, q_att,
                 v_relation_imp, v_relation_sem, v_relation_spa,
                 joint_embedding, classifier, glimpse, fusion, relation_type):
        super().__init__()
        # ReGAT model for each relation
        self.name = "ReGAT_all_%s" % (fusion)
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att

        self.v_relation_imp = v_relation_imp
        self.v_relation_sem = v_relation_sem
        self.v_relation_spa = v_relation_spa

        self.joint_embedding = joint_embedding
        self.classifier = classifier

        # New layers
        self.value_weights = nn.Linear(1024, 1024)
        self.query_weights = nn.Linear(1024, 1024)
        self.score_weights = nn.Linear(1024, 1)
        # self.logit_weights = nn.Linear(1024, 3129)

    def forward(self, v, b, q, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix, labels):
        # RNN
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq)

        # Graph attention
        v_emb_imp = self.v_relation_imp.forward(v, implicit_pos_emb,
                                                q_emb_self_att)
        v_emb_sem = self.v_relation_sem.forward(v, sem_adj_matrix,
                                                q_emb_self_att)
        v_emb_spa = self.v_relation_spa.forward(v, spa_adj_matrix,
                                                q_emb_self_att)

        # BUTD fusion
        if self.fusion == "butd":
            joint_imp, att_imp = self.joint_embedding(v_emb_imp, q_emb)
            joint_sem, att_sem = self.joint_embedding(v_emb_sem, q_emb)
            joint_spa, att_spa = self.joint_embedding(v_emb_spa, q_emb)
        elif self.fusion == "ban":
            joint_imp, att_imp = self.joint_embedding(v_emb_imp, q_emb_seq, b)
            joint_sem, att_sem = self.joint_embedding(v_emb_sem, q_emb_seq, b)
            joint_spa, att_spa = self.joint_embedding(v_emb_spa, q_emb_seq, b)
        else:
            joint_imp, att_imp = self.joint_embedding(v_emb_imp, q_emb_self_att)
            joint_sem, att_sem = self.joint_embedding(v_emb_sem, q_emb_self_att)
            joint_spa, att_spa = self.joint_embedding(v_emb_spa, q_emb_self_att)

        # Ours
        joint = torch.cat((joint_imp.unsqueeze(0), joint_sem.unsqueeze(0), joint_spa.unsqueeze(0)), 0)
        joint_m = self.value_weights(joint)
        q_emb_m = self.query_weights(q_emb)
        joint_q = torch.tanh(joint_m + q_emb_m)
        joint_att_score = self.score_weights(joint_q)
        joint_att = torch.softmax(joint_att_score, 0)  # Weight for each graph relation
        joint_weighted = joint * joint_att
        joint_all = joint_weighted.sum(0)

        # Classifier
        logits = self.classifier(joint_all)

        return logits, att_imp, joint_att


def build_regat_all(dataset, args):
    # @ return values:
    # w_emb, q_emb, q_att, v_relation_imp, v_relation_sem, v_relation_spa,
    # joint_embedding, classifier, glimpse, fusion, relation_type(unused)

    print("Building ReGAT_all model with %s fusion method" %
          (args.fusion))
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600,
                              args.num_hid, 1, False, .0)
    q_att = QuestionSelfAttention(args.num_hid, .2)

    v_relation_sem = ExplicitRelationEncoder(
        dataset.v_dim, args.num_hid, args.relation_dim,
        args.dir_num, args.sem_label_num,
        num_heads=args.num_heads,
        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
        residual_connection=args.residual_connection,
        label_bias=args.label_bias)

    v_relation_spa = ExplicitRelationEncoder(
        dataset.v_dim, args.num_hid, args.relation_dim,
        args.dir_num, args.spa_label_num,
        num_heads=args.num_heads,
        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
        residual_connection=args.residual_connection,
        label_bias=args.label_bias)

    v_relation_imp = ImplicitRelationEncoder(
        dataset.v_dim, args.num_hid, args.relation_dim,
        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
        num_heads=args.num_heads, num_steps=args.num_steps,
        residual_connection=args.residual_connection,
        label_bias=args.label_bias)

    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dataset.num_ans_candidates, 0.5)
    gamma = 0
    if args.fusion == "ban":
        joint_embedding = BAN(args.relation_dim, args.num_hid, args.ban_gamma)
        gamma = args.ban_gamma
    elif args.fusion == "butd":
        joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)
    else:
        joint_embedding = MuTAN(args.relation_dim, args.num_hid,
                                dataset.num_ans_candidates, args.mutan_gamma)
        gamma = args.mutan_gamma
        classifier = None

    return ReGAT_All(dataset, w_emb, q_emb, q_att,
                     v_relation_imp, v_relation_sem, v_relation_spa,
                     joint_embedding, classifier, gamma, args.fusion, args.relation_type)