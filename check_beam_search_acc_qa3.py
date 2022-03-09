import os,sys
import jsonpickle
import pickle
import numpy as np
import torch
import csv

import scipy.stats
import scipy.spatial
import scipy.special
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import sklearn.metrics
np.set_printoptions(precision=2)
np.set_printoptions(precision=4, linewidth=200, suppress=True)

metadata_csv = '/data/share/trojai/trojai-round9-v1-dataset/METADATA.csv'
method_id = 0
base_label_id = 0
target_label_id = 0
gt_results = {}                                                                                                                                                                                         
gt_model_info = {}
mask_fps = []
tp_models = []
all_models = []
for line in open(metadata_csv):
    if len(line.split(',')) == 0:
        continue
    if not line.startswith('id-0000'):
        # head line
        words = line.split(',')

        dataset_id = words.index('source_dataset')
        arch_id = words.index('model_architecture')
        poison_id = words.index('poisoned')
        trigger_id = words.index('trigger.trigger_executor.trigger_text')
        option_id = words.index('trigger.trigger_executor_option')

    else:
        words = line.split(',')
        mname = words[0]
        if words[poison_id] == 'True':
            gt_results[mname] = 1
        else:
            gt_results[mname] = 0

        gt_model_info[mname] = [ gt_results[mname], words[dataset_id], words[arch_id], words[option_id], words[trigger_id], ]

        if words[dataset_id] == 'qa:squad_v2' and words[arch_id] == 'roberta-base':
            all_models.append(mname)

scratch_dir = './scratch_1_1_3_1/'
fns = os.listdir(scratch_dir)

results = {}
results_loss = {}
results_ce_loss = {}
for fn in fns:
    if not ( fn.startswith('id-') and fn.endswith('_features.csv') ):
        continue
    csvreader = csv.reader(open(os.path.join(scratch_dir, fn)), delimiter=',')
    mname = fn.split('_')[0]
    if gt_model_info[mname][1] != 'qa:squad_v2':
        continue
    rows = []
    for row in csvreader:
        # print(len(row), row)
        rows.append(row[:])
    print(rows)
    row = [float(_) for _ in rows[1][1:]]
    results[mname] = [1.0, 1.0] + list(row)[:48]
    results_loss[mname] = list(row)[48:48+72]
    results_ce_loss[mname] = [1.0, 1.0] + list(row)[48+72:48+72+6]

    print(results_loss[mname])

# sys.exit()

missed = []
for mname in all_models:
    if mname not in results.keys():
        missed.append(mname)
print('missed', ' '.join([_[-4:] for _ in missed]), sorted(results.keys()) )

tp = 0
fp = 0
fn = 0
tn = 0
# bound = 0.79
bound = 0.84
all_word_pairs = []
fns = []
tps = []
times = []
fps = []
bacc1 = []
tacc1 = []
bacc2 = []
tacc2 = []
bacc3 = []
tacc3 = []
bacc4 = []
tacc4 = []
bloss1 = []
tloss1 = []
bloss2 = []
tloss2 = []
bloss3 = []
tloss3 = []
bloss4 = []
tloss4 = []
bloss5 = []
tloss5 = []
bloss6 = []
tloss6 = []
bloss7 = []
tloss7 = []
bloss8 = []
tloss8 = []
analyzed = []
xs = []
ys = []
names = []
for mname in sorted(results.keys()):

    if gt_model_info[mname][2] == 'distilbert-base-cased':
        emb_id = 0
    elif gt_model_info[mname][2] == 'google/electra-small-discriminator':
        emb_id = 1
    elif gt_model_info[mname][2] == 'roberta-base':
        emb_id = 2
    else:
        print('error', gt_model_info[mname],)
        sys.exit()

    # x = [emb_id] + list(full_results[mname])
    # xs.append(x)
    # ys.append(gt_results[mname])
    # names.append(mname)

    # if not gt_model_info[mname][2] == 'google/electra-small-discriminator':
    if not gt_model_info[mname][2] == 'roberta-base':
    # if not gt_model_info[mname][2] == 'distilbert-base-cased':
        continue

    # if 'empty' in  gt_model_info[mname][3]:
    # if 'trigger' in  gt_model_info[mname][3]:
    #     continue

    # if not 'subj' in  gt_model_info[mname][1]:
    # # if not 'squad' in  gt_model_info[mname][1]:
    #     continue

    pred = False
    pred1 = False
    pred2 = False

    if len(results[mname]) == 50:
    # if True:
        if gt_model_info[mname][2] == 'roberta-base':
            pred1 = max(results[mname][2:4]) > 0.96 or max(results[mname][26+0:26+2]) > 0.96
            # pred1 = max(results[mname][2:4]) > 0.94

            # if 'squad' in  gt_model_info[mname][1]:
            #     if min(results_ce_loss[mname][:1]) < 0.001:
            #         pred1 = True
            # else:
            #     # if min(results_ce_loss[mname][:1]) < 0.013:
            #     if min(results_ce_loss[mname][:1]) < 0.004:
            #         pred1 = True

            # if min(results_loss[mname][:4]) < 0.59:
            # if min(results_loss[mname][:4]) < 0.29:
            if min(results_loss[mname][:12]) < 0.2 or  min(results_loss[mname][36:36+12]) < 0.2:
                pred1 = True

            if min(results_loss[mname][12:36]) < 7 or  min(results_loss[mname][36+12:36+36]) < 7:
                pred1 = True

            pred2 = max(results[mname][10:26]) > 0.49 or max(results[mname][8+26:24+26]) > 0.49
            # pred2 = max(results[mname][10:]) > 0.66
            # if 'squad' in  gt_model_info[mname][1]:
            #     if min(results_ce_loss[mname][1:]) < 0.002:
            #         pred2 = True
            # else:
            #     if min(results_ce_loss[mname][1:]) < 0.02:
            #         pred2 = True

            if min(results_ce_loss[mname][1:3]) < 0.0015 or min(results_ce_loss[mname][3+1:3+3]) < 0.0015 :
                pred2 = True

            # if results[mname][9] > 0.2:
            #     pred1 = True


            pred = pred1 or pred2

        if gt_results[mname] == 0:
            bacc1.append(max(results[mname][2:4]+results[mname][26+0:26+2]))
        elif 'empty' in gt_model_info[mname][-2]:
            tacc1.append(max(results[mname][2:4]+results[mname][26+0:26+2]))

        if gt_results[mname] == 0:
            bacc2.append(max(results[mname][6:8]+results[mname][26+4:26+6]))
        elif 'empty' in gt_model_info[mname][-2]:
            tacc2.append(max(results[mname][6:8]+results[mname][26+4:26+6]))

        if gt_results[mname] == 0:
            bacc3.append(max(results[mname][10:26]+results[mname][26+8:24+26]))
        elif not 'empty' in gt_model_info[mname][-2]:
            tacc3.append(max(results[mname][10:26]+results[mname][26+8:24+26]))

        if 'squad' in  gt_model_info[mname][1]:
            if gt_results[mname] == 0:
                bloss1.append(min(results_loss[mname][0:12]+results_loss[mname][36+0:36+12]))
            elif 'empty' in gt_model_info[mname][-2]:
                tloss1.append(min(results_loss[mname][0:12]+results_loss[mname][36+0:36+12]))

            if gt_results[mname] == 0:
                bloss2.append(min(results_loss[mname][12:36]+results_loss[mname][36+12:36+36]))
            elif not 'empty' in gt_model_info[mname][-2]:
                tloss2.append(min(results_loss[mname][12:36]+results_loss[mname][36+12:36+36]))

            if gt_results[mname] == 0:
                bloss3.append(min(results_ce_loss[mname][0:1]+results_ce_loss[mname][3+0:3+1]))
            elif 'empty' in gt_model_info[mname][-2]:
                tloss3.append(min(results_ce_loss[mname][0:1]+results_ce_loss[mname][3+0:3+1]))

            if gt_results[mname] == 0:
                bloss4.append(min(results_ce_loss[mname][1:3]+results_ce_loss[mname][3+1:3+3]))
            elif not 'empty' in gt_model_info[mname][-2]:
                tloss4.append(min(results_ce_loss[mname][1:3]+results_ce_loss[mname][3+1:3+3]))

        if gt_results[mname] == 0:
            bacc4.append(results[mname][9])
        else:
            tacc4.append(results[mname][9])

    else:
        print('error', mname, len(results[mname]), results[mname])

    # pred0 = results_pred[mname] > 0.5
    pred0 = False

    # if mname in ['id-00000011', 'id-00000016', 'id-00000023', 'id-00000030', 'id-00000035', 'id-00000045', 'id-00000093', 'id-00000095', 'id-00000111', 'id-00000114']:
    #     pred = True

        

    print(mname, pred1, pred2, pred, pred0, gt_results[mname], len(results[mname]), np.array(results[mname]), '{:.2f}'.format(min(results_loss[mname])), np.array(results_loss[mname]), np.array(results_ce_loss[mname]), gt_model_info[mname], )

    # if gt_model_info[mname][0] == 1:
    #     correct_words = []
    #     for gt_word in gt_model_info[mname][-1].split():
    #         for word in word_pair_dict[mname]:
    #             # if word.strip('Ġ').strip('#').lower() in gt_word.lower():
    #             if word.strip('Ġ').strip('#').lower() in gt_word.lower() and len(gt_word) - len(word) <3:
    #                 correct_words.append(word)
    #     print('correct words', correct_words)

    # pred = pred0

    if gt_results[mname] == 1 and pred:
        tp += 1
        tps.append(mname)
    elif gt_results[mname] == 0 and pred:
        fp += 1
        fps.append(mname)
    elif gt_results[mname] == 1 and not pred:
        fn += 1
        fns.append(mname)
    elif gt_results[mname] == 0 and not pred:
        tn += 1
    # times.append(time_dict[mname])
    analyzed.append(mname)

# for mname in early_determined:
#     if gt_results[mname] == 1:
#         tp += 1
#         tps.append(mname)
#     elif gt_results[mname] == 0:
#         fp += 1
#         fps.append(mname)


print('tp', tp, 'fp', fp, 'fn', fn, 'tn', tn, )
print('fns', fns)
print('fps', fps)
print('fns', ' '.join([_[-4:] for _ in fns]))
print('fps', ' '.join([_[-4:] for _ in fps]))
print('tps', ' '.join([_[-4:] for _ in tps]))
# print('all time', sum(times), sum(times)/len(times))
for mname in fns:
    print('fn', mname, gt_model_info[mname])
for mname in fps:
    print('fp', mname, gt_model_info[mname])
print('benign model acc', np.array(sorted(bacc1)))
print('trojan model acc', np.array(sorted(tacc1)))
print('benign model acc', np.array(sorted(bacc2)))
print('trojan model acc', np.array(sorted(tacc2)))
print('benign model acc', np.array(sorted(bacc3)))
print('trojan model acc', np.array(sorted(tacc3)))
print('benign model acc', np.array(sorted(bacc4)))
print('trojan model acc', np.array(sorted(tacc4)))
print('benign model loss', np.array(sorted(bloss1)))
print('trojan model loss', np.array(sorted(tloss1)))
print('benign model loss', np.array(sorted(bloss2)))
print('trojan model loss', np.array(sorted(tloss2)))
print('benign model squad loss', np.array(sorted(bloss3)))
print('trojan model squad loss', np.array(sorted(tloss3)))
print('benign model subjqa loss', np.array(sorted(bloss7)))
print('trojan model subjqa loss', np.array(sorted(tloss7)))
print('benign model squad loss', np.array(sorted(bloss4)))
print('trojan model squad loss', np.array(sorted(tloss4)))
print('benign model subjqa loss', np.array(sorted(bloss8)))
print('trojan model subjqa loss', np.array(sorted(tloss8)))

# for mname in analyzed:

#     if gt_model_info[mname][0] == 1:
#         correct_words = []
#         for gt_word in gt_model_info[mname][-1].split():
#             for word in word_pair_dict[mname]:
#                 # if word.strip('Ġ').strip('#').lower() in gt_word.lower():
#                 if word.strip('Ġ').strip('#').lower() in gt_word.lower() and len(word.strip('Ġ').strip('#').lower()) >1:
#                 # if word.strip('Ġ').strip('#').lower() in gt_word.lower() and len(gt_word) - len(word) <5:
#                     correct_words.append(word)
#         print('correct words', mname, correct_words, gt_model_info[mname])

# # sys.exit()

# xs = np.array(xs)
# ys = np.array(ys)
# print('xs', xs.shape, ys.shape)
# Xs = xs

# ne = 2000
# md = 2


# train_accs = []
# test_accs = []
# roc_aucs = []
# ce_losses = []
# kf = KFold(n_splits=5, shuffle=True)
# for train_index, test_index in kf.split(Xs):
#     train_X, test_X = Xs[train_index], Xs[test_index]
#     # train_X, test_X = X_normalized[train_index], X_normalized[test_index]
#     train_y, test_y = ys[train_index], ys[test_index]

#     tnames = []
#     for i in test_index:
#         tnames.append(names[i])

#     # ir_models = []
#     # train_X2 = np.zeros(train_X.shape)
#     # for i in range(train_X.shape[1]):
#     #     ir_model = IsotonicRegression(out_of_bounds='clip')
#     #     pcal = ir_model.fit_transform(train_X[:,i], train_y)
#     #     train_X2[:,i] = pcal
#     #     ir_models.append(ir_model)
#     # train_X = train_X2
        
#     cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False)
#     # cls = RandomForestClassifier(n_estimators=ne, max_depth=md)
#     cls.fit(train_X, train_y)

#     # reg = LinearRegression().fit(train_X, train_y)
#     # cls = reg
#     # cls = SVC(gamma='auto')
#     # cls.fit(train_X, train_y)

#     preds = cls.predict(train_X)

#     # print('preds', preds)
#     fp = 0
#     tp = 0
#     fn = 0
#     tn = 0
#     for i in range(train_y.shape[0]):
#         # print(train_X[i], preds[i])
#         if preds[i] > 0.5 and train_y[i] == 1:
#             tp += 1
#         elif preds[i] > 0.5 and train_y[i] == 0:
#             fp += 1
#         elif preds[i] <= 0.5 and train_y[i] == 0:
#             tn += 1
#         elif preds[i] <= 0.5 and train_y[i] == 1:
#             fn += 1
#     print('train', 'tp', tp, 'fp', fp, 'fn', fn, 'tn', tn, 'acc', (tp+tn)/float(tp+fp+fn+tn))
#     train_accs.append((tp+tn)/float(tp+fp+fn+tn))

#     # test_X2 = np.zeros(test_X.shape)
#     # for i in range(train_X.shape[1]):
#     #     ir_model = ir_models[i]
#     #     pcal = ir_model.transform(test_X[:,i])
#     #     test_X2[:,i] = pcal
#     # test_X = test_X2
    
#     preds = cls.predict(test_X)
    
#     fp = 0
#     tp = 0
#     fn = 0
#     tn = 0
#     tps = []
#     fps = []
#     fns = []
#     for i in range(test_y.shape[0]):
#         if preds[i] > 0.5 and test_y[i] == 1:
#             tp += 1
#             # tps.append(tnames[i])
#         elif preds[i] > 0.5 and test_y[i] == 0:
#             fp += 1
#             # fps.append(tnames[i])
#             print('fp', tnames[i]), test_X[i]
#         elif preds[i] <= 0.5 and test_y[i] == 0:
#             tn += 1
#         elif preds[i]<= 0.5 and test_y[i] == 1:
#             fn += 1
#             # fns.append(tnames[i])
#             print('fn', tnames[i])
#     print('test', 'tp', tp, 'fp', fp, 'fn', fn, 'tn', tn, 'acc', (tp+tn)/float(tp+fp+fn+tn))
#     test_accs.append((tp+tn)/float(tp+fp+fn+tn))

#     confs = cls.predict_proba(test_X)[:,1]

#     confs = np.clip(confs, 0.025, 0.975)
#     print('before confs', confs)

#     # iso_reg = IsotonicRegression(out_of_bounds='clip')
#     # iso_reg.fit(cls.predict_proba(train_X)[:,1], train_y)
#     # confs = iso_reg.transform(cls.predict_proba(test_X)[:,1])

#     # lr_reg = LogisticRegression(C=100, max_iter=10000, tol=1e-4)
#     lr_reg = LogisticRegression(max_iter=10000, tol=1e-4)
#     # lr_reg.fit(cls.predict_proba(train_X), train_y)
#     # confs = lr_reg.predict_proba(cls.predict_proba(test_X))[:,1]

#     lr_reg.fit(np.concatenate([train_X, cls.predict_proba(train_X)], axis=1) , train_y)
#     confs = lr_reg.predict_proba( np.concatenate([test_X, cls.predict_proba(test_X)], axis=1) )[:,1]

#     # clf = CalibratedClassifierCV(cls, method='isotonic')
#     # clf.fit(train_X, train_y)
#     # confs = clf.predict_proba(test_X)[:,1]

#     # confs = np.clip(confs, 0.05, 0.95)
#     confs = np.clip(confs, 0.025, 0.975)
#     print('after confs', confs)


#     print('test_y',test_y.shape)
#     roc_auc = sklearn.metrics.roc_auc_score(test_y, confs)
#     celoss  = sklearn.metrics.log_loss(test_y, confs)
#     print('roc_auc', roc_auc, 'celoss', celoss)
#     roc_aucs.append(roc_auc)
#     ce_losses.append(celoss)
#     # print('tps', tps)
#     # print('fps', fps)
#     # print('fns', fns)
# train_accs = np.array(train_accs)
# test_accs  = np.array(test_accs)
# roc_aucs = np.array(roc_aucs)
# ce_losses  = np.array(ce_losses)
# print('train accs', np.mean(train_accs), np.var(train_accs))
# print('test accs', np.mean(test_accs), np.var(test_accs))
# print('test roc_aucs', np.mean(roc_aucs), np.var(roc_aucs))
# print('test ce_losses', np.mean(ce_losses), np.var(ce_losses))

# # cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False)
# # cls.fit(Xs, ys)
# # lr_reg = LogisticRegression(max_iter=10000, tol=1e-4)
# # lr_reg.fit(np.concatenate([xs, cls.predict_proba(xs)], axis=1) , ys)

# cls, lr_reg = pickle.load(open('./contain_abs_1_1_4/learned_parameters/rf_lr_qa1.pkl', 'rb'))

# confs = lr_reg.predict_proba( np.concatenate([xs, cls.predict_proba(xs)], axis=1) )[:,1]
# confs = np.clip(confs, 0.025, 0.975)
# print('after confs', confs.shape, confs)
# roc_auc = sklearn.metrics.roc_auc_score(ys, confs)
# celoss  = sklearn.metrics.log_loss(ys, confs)
# print('overall roc_auc', roc_auc, 'celoss', celoss)

# # pickle.dump((cls, lr_reg), open('./rf_lr_qa.pkl', 'wb'))
