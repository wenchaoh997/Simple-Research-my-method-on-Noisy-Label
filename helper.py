"""
Simulate my noisy dataset
and set a SEED to make it unvaried
"""

import random

def noisyHelper(raw, noisyRate, SEED=924, N_type='symmetry'):
    noise = int(len(raw) * noisyRate)
    random.seed(SEED)
    modInx = random.sample(range(len(raw)), noise)

    classLen = len(set(list(raw.targets)))
    # print(list(raw.targets[:100]))
    # print(set(list(raw.targets[:100].tolist())))
    if N_type == 'symmetry':
        print('Num', classLen)
        for i in modInx:
            tmp = list(range(classLen))
            label = int(raw.targets[i])
            del(tmp[label])
            random.seed(i)
            noise = random.choice(tmp)
            raw.targets[i] = noise
    elif N_type == 'pair':
        print('Num', classLen)
        tmp = list(range(classLen))
        for i in modInx:
            label = int(raw.targets[i])
            random.seed(i)
            noise = tmp[(label+1) % classLen]
            raw.targets[i] = noise

    return raw, modInx
