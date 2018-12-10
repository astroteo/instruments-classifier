
import librosa as lb
import numpy as np

def get_features_d(song):

    if song.shape[0]==2:

        song = np.sum(song/ 2, axis=0)

    else:
        song = song
    x_raw = song
    R = lb.feature.rmse(x_raw)[0]
    DR = lb.feature.delta(R)[0]
    #DDR = lb.feature.delta(DR)[0]

    Z = lb.feature.zero_crossing_rate(x_raw)[0]
    DZ = lb.feature.delta(Z)[0]
    #DDZ = lb.feature.delta(DZ)[0]

    S_mfcc = lb.feature.mfcc(y=song, n_mfcc=10,sr=44100)
    DS_mfcc = lb.feature.delta(S_mfcc)
    #DDS_mfcc = lb.feature.delta(DS_mfcc)

    mfcc = np.mean(S_mfcc, axis=1)
    mfcc_s = np.std(S_mfcc, axis=1)
    d_mfcc = np.mean(DS_mfcc, axis=1)
    d_mfcc_s = np.std(DS_mfcc, axis=1)

    energy_dic = {
        'rmse_av': np.mean(R),
        'zcr_av': np.mean(Z),
        'rmse_std': np.std(R),
        'zcr_std': np.std(Z),
        'd_rmse_av': np.mean(DR),
        'd_zcr_av': np.mean(DZ),
        'd_rmse_std': np.std(DR),
        'd_zcr_std': np.std(DZ)
    }

    mfcc_dic = {}
    mfcc_dic_s = {}
    for i in range(len(mfcc)):
        mfcc_key = ("mfcc_%04d" % i)
        mfcc_dic[mfcc_key] = mfcc[i]
        mfcc_s_key = ("mfcc_s_%04d" % i)
        mfcc_dic[mfcc_s_key] = mfcc_s[i]

    d_mfcc_dic = {}
    d_mfcc_dic_s = {}
    for i in range(len(d_mfcc)):
        d_mfcc_key = "d_mfcc_{}".format(i)
        d_mfcc_dic[d_mfcc_key] = d_mfcc[i]
        d_mfcc_s_key = ("d_mfcc_s_{}".format(i))
        d_mfcc_dic[d_mfcc_s_key] = d_mfcc_s[i]



    super_dict = {}
    dict_list = [energy_dic, mfcc_dic, mfcc_dic_s, d_mfcc_dic, d_mfcc_dic_s]

    for d in dict_list:
        for k, v in d.items():
            super_dict[k] = v

    sorted_keys = sorted(super_dict.keys())

    super_dict_sorted ={}

    for k in sorted_keys:
        super_dict_sorted[k]=super_dict[k]



    feats_list = [v for v in super_dict_sorted.values()]

    feats = np.array(feats_list)
    return feats
