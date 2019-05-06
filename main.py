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

def cn(cut):
    return {"25": r"\#1", "45": r"\#2"}[cut]


settings = [
    {"cut": "25", "horizon": 50},
    {"cut": "25", "horizon": 100},
    {"cut": "45", "horizon": 50},
    {"cut": "45", "horizon": 100},
]

# settings = [{"cut": "25", "horizon": x} for x in range(20, 1000, 20)]



overall_average_mae = []
overall_average_maef = []
overall_average_maerm = []



for setting in settings:

    CUT = setting["cut"]
    PATH = "data/data{}csv.csv".format(CUT)

    # get and rename channels
    data = pd.read_csv(PATH)
    orig_channels = [_ for _ in data.keys()if "TM" in _ and not "TM45LD" in _]
    channels = []
    for idx, ch in enumerate(orig_channels):
        ch_name = "CH" + str(idx)
        data[ch_name] = data[ch]
        channels.append(ch_name)


    # normalization
    z_score_params = {}
    for ch in channels:
        data["orig_{}".format(ch)] = data[ch]
        mean = data[ch].mean()
        std = data[ch].std(ddof=0)
        data[ch] = (data[ch] - mean) / std / 3.
        z_score_params[ch] = (mean, std)


    horizon = setting["horizon"]
    step = horizon
    history = horizon + (10 * step)

    windows = list(range(horizon, history, step))

    data = data.loc[max(windows):len(data)-max(windows)]

    learning_curve = {}
    weights_stats = []
    table_body = []
    averages = []

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

        IMP_MAE = (1 - (MAE / min(MAE_RM, MAE_F))) * 100

        """
        Comparison table
        """
        out_mae = make_bold(MAE) if MAE < MAE_F and MAE < MAE_RM else make_normal(MAE)
        out_maef = make_bold(MAE_F) if MAE_F < MAE and MAE_F < MAE_RM else make_normal(MAE_F)
        out_maerm = make_bold(MAE_RM) if MAE_RM < MAE and MAE_RM < MAE_F else make_normal(MAE_RM)

        out_mse = make_bold(MSE) if MSE < MSE_F and MSE < MSE_RM else make_normal(MSE)
        out_msef = make_bold(MSE_F) if MSE_F < MSE and MSE_F < MSE_RM else make_normal(MSE_F)
        out_mserm = make_bold(MSE_RM) if MSE_RM < MSE and MSE_RM < MSE_F else make_normal(MSE_RM)

        imp_mae = make_normal(IMP_MAE, r=2, fill=2)

        line = " & ".join([ch, out_mae, out_mse, out_maef, out_msef, out_maerm, out_mserm, imp_mae]) + r"\\"
        table_body.append(line)
        averages.append([MAE, MSE, MAE_F, MSE_F, MAE_RM, MSE_RM, IMP_MAE])

        # print(line)

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


    """
    Print table
    """
    # header
    print("\n"*1)
    print(r"""
    \begin{table}[H]
    \caption{The detail results for for engine temperature field """ + cn(str(CUT)) +  r""" and prediction horizon of """ + str(horizon) + r""" samples. The best result in every channel is highlighted in bold. The MAE is in degrees Celsius.}
    \centering
    \label{tab:results""" + str(CUT) + "_" + str(horizon) + r"""}
    \begin{tabular}{c|cc|cc|cc|c}
    \toprule
    \textbf{Channel} & \multicolumn{2}{c}{\textbf{Proposed method}} & \multicolumn{2}{c}{\textbf{Reference 1}}	& \multicolumn{2}{c}{\textbf{Reference 2}} & \textbf{MAE} \\
    \textbf{Name} & \multicolumn{2}{c}{\textbf{NLMS filtering}} & \multicolumn{2}{c}{\textbf{Follower}} & \multicolumn{2}{c}{\textbf{Rolling mean}} & \textbf{Improvement} \\
    \midrule
     & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{[\%]}\\
    \midrule
    """)
    # table
    for line in table_body:
        print(line)
    # averages
    av_data = [
        [_[0] for _ in averages],
        [_[1] for _ in averages],
        [_[2] for _ in averages],
        [_[3] for _ in averages],
        [_[4] for _ in averages],
        [_[5] for _ in averages],
        [_[6] for _ in averages],
    ]
    av_avg_num = [np.mean(_) for _ in av_data]
    overall_average_mae.append(av_avg_num[0])
    overall_average_maef.append(av_avg_num[2])
    overall_average_maerm.append(av_avg_num[4])
    av_avg = [make_bold(_) for _ in av_avg_num]
    av_line = " & ".join([r"\textbf{Average}", ] + av_avg) + r"\\"
    print(r"\midrule")
    print(av_line)
    # bottom
    print(r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """)
    print("\n"*1)


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



# plt.plot([s["horizon"] for s in settings], overall_average_mae, label="NLMS")
# plt.plot([s["horizon"] for s in settings], overall_average_maef, label="follower")
# plt.plot([s["horizon"] for s in settings], overall_average_maerm, label="rolling mean")
# plt.legend()
# plt.show()