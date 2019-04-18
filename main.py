import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import padasip as pa

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def make_normal(a, r=5, fill=7):
    return str(np.round(a, r)).ljust(fill, '0')

def make_bold(a):
    a = make_normal(a)
    return r"\textbf{" + a + "}"

CUT = "25"
PATH = "data/data{}csv.csv".format(CUT)


data = pd.read_csv(PATH)
channels = [_ for _ in data.keys()if "TM" in _]

# normalization
z_score_params = {}
for ch in channels:
    data["orig_{}".format(ch)] = data[ch]
    mean = data[ch].mean()
    std = data[ch].std(ddof=0)
    data[ch] = (data[ch] - mean) / std / 3.
    z_score_params[ch] = (mean, std)


# h = 50, s = 50, 10*step, mu=0.01
# 10, 10, 5*step, 0.01
# 5, 5, 5*step, 0.01
horizon = 50
step = horizon
history = horizon + (10 * step)

windows = list(range(horizon, history, step))

data = data.loc[max(windows):len(data)-max(windows)]

results = []

learning_curve = {}
weights_stats = []

for chidx, ch in enumerate(channels[:]):

    # make the shifted channels
    target_keys = []
    for win in windows:
        name = "{}_{}".format(ch, win)
        data[name] = data[ch].shift(win)     
        target_keys.append(name)

    # make rolling mean
    data["{}_mean".format(name)] = data[ch].rolling(100).mean().shift(horizon)

    # prepare sample
    sample = data[history:-history-1]

    # filtration
    filt = pa.filters.FilterNLMS(n=len(target_keys), mu=0.001, w="zeros")
    # filt = pa.filters.FilterRLS(n=len(target_keys), mu=0.99, w="zeros")

    # pretrain
    x = sample[target_keys].values
    d = sample[ch].values
    ntrain = 20000
    learning_curve[ch] = []
    for _ in range(5):
        y, e, w = filt.run(d[:ntrain], x[:ntrain])
        learning_curve[ch].append(np.dot(e,e))

    # real run
    y = np.zeros(len(d))
    e = np.zeros(len(d))
    w = np.zeros((len(d), len(target_keys)))
    for idx in range(len(d)):
        y[idx] = filt.predict(x[idx])
        e[idx] = y[idx] - d[idx]
        w[idx] = filt.w
        if idx > history:
            filt.adapt(d[idx-history], x[idx-history])


    shift_name = "{}_{}".format(ch, windows[0])

    # denormalize
    mean = z_score_params[ch][0]
    std = z_score_params[ch][1]
    sample_origin = sample[ch] * std + mean
    sample_shifted = sample[shift_name] * std + mean
    sample_rm = sample["{}_mean".format(name)] * std + mean
    e = (y[:] * std + mean) - (d[:] * std + mean)

    # calculate erros (MAE, MSE)
    MEASUREMENT_SAMPLE = 20000
    MSE = np.dot(e[-MEASUREMENT_SAMPLE:],e[-MEASUREMENT_SAMPLE:]) / MEASUREMENT_SAMPLE
    MAE = abs(e[-MEASUREMENT_SAMPLE:]).mean()
    
    shift_ch = (sample_shifted - sample_origin)[-MEASUREMENT_SAMPLE:]
    MSE_F = np.dot(shift_ch, shift_ch) / MEASUREMENT_SAMPLE
    MAE_F = shift_ch.abs().mean()
    
    rolling_mean = (sample_rm - sample_origin)[-MEASUREMENT_SAMPLE:]
    MSE_RM = np.dot(rolling_mean, rolling_mean) / MEASUREMENT_SAMPLE
    MAE_RM = rolling_mean.abs().mean()


    """
    Comparison table
    """
    out_mae = make_bold(MAE) if MAE < MAE_F and MAE < MAE_RM else make_normal(MAE)
    out_maef = make_bold(MAE_F) if MAE_F < MAE and MAE_F < MAE_RM else make_normal(MAE_F)
    out_maerm = make_bold(MAE_RM) if MAE_RM < MAE and MAE_RM < MAE_F else make_normal(MAE_RM)

    out_mse = make_bold(MSE) if MSE < MSE_F and MSE < MSE_RM else make_normal(MSE)
    out_msef = make_bold(MSE_F) if MSE_F < MSE and MSE_F < MSE_RM else make_normal(MSE_F)
    out_mserm = make_bold(MSE_RM) if MSE_RM < MSE and MSE_RM < MSE_F else make_normal(MSE_RM)

    imp_mae = make_normal((1 - (MAE / min(MAE_RM, MAE_F))) * 100, r=2, fill=2)

    line = " & ".join([ch, out_mae, out_mse, out_maef, out_msef, out_maerm, out_mserm, imp_mae]) + r"\\"

    print(line)


    error = abs(e)

    skip = 0

    """
    DETAIL look at the output
    """
    # LW = 1
    # plt.plot(sample[ch].values[skip:], "k-", linewidth=LW*2, label="original target")
    # plt.plot(y[skip:], "k-", linewidth=LW, label="Proposed- NLMS")
    # plt.plot(sample["{}_mean".format(name)].values[skip:], "k--", linewidth=LW, label="Ref2 - rolling mean")
    # plt.plot(sample[shift_name].values[skip:], "k:", linewidth=LW, label="Ref1 - follower")
    # plt.ylabel("Normalized output [-]")
    # plt.xlabel("k [-]")
    # plt.xlim(29500, 30100)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()

    
    """
    WEIGHTS adaptation
    """
    # plt.plot(w, "k")
    # plt.ylabel("Weights [-]")
    # plt.xlabel("k [-]")
    # plt.tight_layout()
    # plt.show()


    """
    WEIGHTS stats (mean, std)
    """
    weights_stats.append({
        "name": ch,
        "mean": w.mean(axis=0),
        "std": w.std(axis=0),
    })


    # plt.subplot(311)
    # plt.title(ch)
    # plt.plot(error[skip:], "r", label="error")
    # plt.subplot(312)
    # plt.plot(y[skip:], "g", label="reconstruction")
    # plt.plot(sample[ch].values[skip:], "b", label="real")
    # plt.plot(sample["{}_mean".format(name)].values[skip:], "r", label="rolling_mean")
    # plt.plot(sample[shift_name].values[skip:], "k", label="follower")
    # plt.subplot(313)
    # plt.plot(w[skip:])
    # plt.tight_layout()
    # plt.show()





    # mean = z_score_params[ch][0]
    # std = z_score_params[ch][1]
    # plt.plot(y[skip:] * std + mean, "g", label="reconstruction")
    # plt.plot(data["orig_{}".format(ch)].values[skip:], "b", label="real")
    # plt.show()


    # plt.subplot(311)
    # plt.title(ch + " vs " + ref)
    # plt.plot(e[skip:], "r", label="error")
    # plt.subplot(312)
    # plt.plot(y[skip:], "g", label="reconstruction")
    # # plt.plot(goal.values[skip:], "b", label="real")
    # plt.plot(data[ch].values[skip:], "b", label="real")
    # plt.subplot(313)
    # plt.plot(w[skip:])
    # plt.tight_layout()
    # plt.show()





#     plt.plot(y[skip:]+3*chidx, "g", label="reconstruction")
#     plt.plot(data[ch].values[skip:]+3*chidx, "b", label="real")

# plt.show()


"""
Plot of learnings
"""
filename = "figs/learning{}.png".format(CUT)
plt.figure(figsize=(6,3))
for name, vals in learning_curve.items():
    plt.plot(np.arange(1,len(vals)+1), vals, "k")
plt.xlabel("Training epoch")
plt.xticks(np.arange(1,len(vals)+1), np.arange(1,len(vals)+1))
plt.ylabel("Sum of squared error")
plt.tight_layout()
plt.grid()
plt.savefig(filename, dpi=300, bbox_inches=None)
plt.clf()
plt.close()


"""
Plot of weights
"""
filename = "figs/weights_distribution{}.png".format(CUT)
plt.figure(figsize=(6,4))
for idx, item in enumerate(weights_stats):
    for n in range(len(item["mean"])):
        hp = item["mean"][n] + item["std"][n]
        lp = item["mean"][n] - item["std"][n]
        plt.plot([idx, idx], [lp, hp], "k")
        plt.plot([idx, ], [item["mean"][n], ], "ok")
plt.xticks(np.arange(len(weights_stats)), [_["name"] for _ in weights_stats], rotation='vertical')
plt.grid()
plt.tight_layout()
plt.savefig(filename, dpi=300, bbox_inches=None)
plt.clf()
plt.close()
