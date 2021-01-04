# %%
import numpy as np
import pandas as pd

# %%


def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=100, fill='█', printEnd=""):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' %
          (prefix, bar, percent, suffix), end=printEnd, flush=True)
    # Print New Line on Complete
    if iteration == total:
        print()

# %%


def read_file_3lines(file, start_user):
    user_ids = []
    skill_ids = []
    correct = []
    with open(file, "r") as f:
        line = f.readline()
        cnt = 0
        user_id = start_user
        try:
            num_responses = int(line)
        except:
            print('Error')
        user_ids += [user_id]*num_responses
        while line:
            line = f.readline()
            if line == "":
                break
            cnt += 1
            if cnt % 3 == 0:
                user_id += 1
                num_responses = int(line)
                user_ids += [user_id]*num_responses
            elif cnt % 3 == 1:
                skill_ids += line.replace("\n", "").split(",")
            elif cnt % 3 == 2:
                correct += line.replace("\n", "").split(",")
        user_ids = np.reshape(np.array(user_ids), [-1, 1])
        num_unique_users = np.unique(user_ids[:, 0]).shape[0]
        skill_ids = np.reshape(np.array(skill_ids).astype(int), [-1, 1])
        correct = np.reshape(np.array(correct).astype(int), [-1, 1])
        idx = np.reshape((correct == 0) + (correct == 1), [-1])
        data = np.hstack((user_ids[idx], skill_ids[idx], correct[idx]))
        return data, num_unique_users

# %%


def read_file(file):
    user_ids = []
    skill_ids = []
    correct = []
    unique_users = []

    dataset = pd.read_csv(file, sep=',', header=0)
    dataset.sort_values(by=["user_id"], inplace=True)

    user_ids = dataset["user_id"].astype(int)
    unique_users = np.unique(user_ids)
    num_unique_users = unique_users.shape[0]

    for id in unique_users:
        user_data = dataset.query('user_id == @id')
        user_skill_ids = user_data["skill_id"].astype(int)
        user_answers = user_data["correct"].astype(int)
        skill_ids = np.concatenate((skill_ids, user_skill_ids)).astype(int)
        correct = np.concatenate((correct, user_answers)).astype(int)

    user_ids = np.reshape(np.array(user_ids), [-1, 1])
    skill_ids = np.reshape(np.array(skill_ids), [-1, 1])
    correct = np.reshape(np.array(correct), [-1, 1])
    idx = np.reshape((correct == 0) + (correct == 1), [-1])

    data = np.hstack((user_ids[idx], skill_ids[idx], correct[idx]))
    return data, num_unique_users
