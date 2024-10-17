# Bailando
import numpy as np
import os
from utils.features.kinetic import extract_kinetic_features
from utils.features.manual_new import extract_manual_features
from scipy import linalg

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def calc_metrics(gt_root, pred_root):

    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []

    pred_features_k = [np.load(os.path.join(pred_root, 'kinetic_features', npy)) for npy in os.listdir(os.path.join(pred_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(pred_root, 'manual_features_new', npy)) for npy in os.listdir(os.path.join(pred_root, 'manual_features_new'))]

    gt_freatures_k = [np.load(os.path.join(gt_root, 'kinetic_features', npy)) for npy in os.listdir(os.path.join(gt_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_root, 'manual_features_new', npy)) for npy in os.listdir(os.path.join(gt_root, 'manual_features_new'))]

    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k)
    gt_freatures_m = np.stack(gt_freatures_m)

    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m)

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)

    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)

    metrics = {'fid_k': fid_k, 'fid_m': fid_m, 'div_k': div_k, 'div_m' : div_m, 'div_k_gt': div_k_gt, 'div_m_gt': div_m_gt}
    # metrics = {'fid_k': fid_k, 'fid_m': fid_m}
    return metrics

def calc_fid(pred_features, gt_features):

    # mu_gen = np.mean(pred_features, axis=0)
    # sigma_gen = np.cov(pred_features, rowvar=False)

    # mu_gt = np.mean(gt_features, axis=0)
    # sigma_gt = np.cov(gt_features, rowvar=False)

    # mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    # diff = mu1 - mu2
    # eps = 1e-5
    # # Product might be almost singular
    # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # if not np.isfinite(covmean).all():
    #     msg = ('fid calculation produces singular product; '
    #            'adding %s to diagonal of cov estimates') % eps
    #     print(msg)
    #     offset = np.eye(sigma1.shape[0]) * eps
    #     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # # Numerical error might give slight imaginary component
    # if np.iscomplexobj(covmean):
    #     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
    #         m = np.max(np.abs(covmean.imag))
    #         # raise ValueError('Imaginary component {}'.format(m))
    #         covmean = covmean.real

    # tr_covmean = np.trace(covmean)

    # return (diff.dot(diff) + np.trace(sigma1)
    #         + np.trace(sigma2) - 2 * tr_covmean)

    mu1 = np.mean(pred_features, axis=0)
    sigma1 = np.cov(pred_features, rowvar=False)
    mu2 = np.mean(gt_features, axis=0)
    sigma2 = np.cov(gt_features, rowvar=False)

    ssd = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssd + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_save_feats(root):
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))

    npy_path = os.path.join(root, 'npy')

    for npy in os.listdir(npy_path):
        joints3d = np.load(os.path.join(npy_path, npy), allow_pickle=True).item()['position'] # [nframes, 24, 3]
        np.save(os.path.join(root, 'kinetic_features', npy), extract_kinetic_features(joints3d.reshape(-1, 24, 3)))
        np.save(os.path.join(root, 'manual_features_new', npy), extract_manual_features(joints3d.reshape(-1, 24, 3)))

if __name__ == '__main__':
    gt_root = 'data/aist_features_zero_start'
    pred_root = 'data/aistpp_evaluation/pred'
    calc_save_feats(pred_root)

    result_path = 'data/aistpp_evaluation/metrics.txt'
    result = calc_metrics(gt_root, pred_root)
    with open(result_path, 'w') as f:
        f.write(str(result))
        f.write('\n')

    print(result)