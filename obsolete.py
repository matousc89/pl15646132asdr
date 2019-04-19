import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import padasip as pa

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

PATH = "artificial data/data25csv.csv"

data = pd.read_csv(PATH)

channels = [_ for _ in data.keys()if "TM" in _]


# normalization
z_score_params = {}
for ch in channels:
    data["orig_{}".format(ch)] = data[ch]
    mean = data[ch].mean()
    std = data[ch].std(ddof=0) * 3.
    data[ch] = (data[ch] - mean) / std
    z_score_params[ch] = (mean, std)


# for ch in channels:
#     plt.plot(data["TIME"], data[ch])
# plt.show()


refs = [channels[0], channels[1], channels[-1], channels[-2]]
windows = [0, 5, 10, 15, 20]#, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 105]
target_keys = []
for win in windows:
    for ref in refs:
        name = "{}_{}".format(ref, win)
        data[name] = data[ref].shift(-win) 
        target_keys.append(name)


print(target_keys)

data = data.loc[200:len(data)-200]

for chidx, ch in enumerate(channels[2:-2]):
    print(ch)
    goal = data[ch]

    # make the shuffled adaptation
    f = pa.filters.FilterNLMS(n=len(target_keys), mu=0.1, w="random")
    ntrain = (len(data) // 3) * 2
    for idx in range(100):
        f.mu *= 0.98
        x = data[target_keys].values[ntrain:]
        d = goal.values[ntrain:]
        x, d = unison_shuffled_copies(x, d) # shuffle!
        y, e, w = f.run(d, x)
        print(idx, f.mu, np.dot(e,e))


    # make the real simulation with no adaptation
    x = data[target_keys].values[:]
    y = np.dot(x, w[-1])


    # plt.plot(data[ch][e > 0.2].values, data[ref][e > 0.2].values, "r.")
    # plt.plot(data[ch][e < -0.2].values, data[ref][e < -0.2].values, "b.")
    # plt.show()

    # plt.plot(data[channels[ref]], data[ch], ".")
    # plt.show()



    skip = 0


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





    plt.plot(y[skip:]+3*chidx, "g", label="reconstruction")
    plt.plot(data[ch].values[skip:]+3*chidx, "b", label="real")

plt.show()

