#!/usr/bin/env python
# coding: utf-8

# ### Assistments2009-corrected data
# * Read data: `assistment2009_corrected.csv`
# * Read skill names: `skill_names_corrected.csv`
# * Read skill name embeddings: `skill_name_embeddings_corrected.csv`

# In[1]:

import random
import math
import pandas as pd
import numpy as np
from os import path, getcwd
import sys
from Utils import printProgressBar

# rootdir needs to change depending on user
rootdir = 'C:/Users/Vasilis Nikiforidis/Desktop/Thesis/MyCode/'
sys.path.append(rootdir)

data_set = "assistment_2009_corrected"
full_set = "skill_builder_data_corrected_collapsed.csv"
# In[2]:

# ### Read Assistments data to DataFrame `data`
# Clean skill names and assign skill and student ids.

"""
Create new data split
"""

# Read skill names
sknames_file = path.join(
    rootdir, "embeddings/" + data_set + "/skill_names_corrected.csv")
skill_names = pd.read_csv(sknames_file, sep=',', header=None).values
num_skills = skill_names.shape[0]

# Read complete data set
full_data_file = path.join(rootdir, "data/" + full_set)

# Read assistments data
print("Reading and preprocessing {} ...".format(full_data_file), end="")
data = pd.read_csv(full_data_file, sep=',', header=0, encoding='latin-1')
chosen_columns = ["user_id", "skill_id", "skill_name", "correct"]
data = data[chosen_columns]
data = data.dropna()

# Transform student_ids to student-indices, ie. values from 1 to max
student_ids = data.user_id.unique()
num_students = len(student_ids)
student_dict = {student_ids[i]: (i+1) for i in range(student_ids.shape[0])}

# Transform student_ids to student-indices, ie. values from 1 to max
i = 0
printProgressBar(0, num_students, prefix='Progress:',
                 suffix='Complete', length=50)
for old_id, new_id in student_dict.items():
    i += 1
    data.loc[data['user_id'] == old_id, 'user_id'] = -new_id
    printProgressBar(i+1, num_students, prefix='Pass 1:',
                     suffix='Complete', length=50)
printProgressBar(0, num_students, prefix='Progress:',
                 suffix='Complete', length=50)
for stud_id in range(1, num_students+1):
    data.loc[data['user_id'] == -stud_id, 'user_id'] = stud_id
    printProgressBar(i+1, num_students, prefix='Pass 2:',
                     suffix='Complete', length=50)

unused_skills = []
skill_names_to_change = {
    "Effect of Changing Dimensions of a Shape Proportionally": "Effect of Changing Dimensions of a Shape Prportionally",
    "Understanding concept of probabilities": "D.4.8-understanding-concept-of-probabilities",
    "Order of Operations addition subtraction division multiplication parentheses positive reals": "Order of Operations +,-,/,* () positive reals",
    "Angles Obtuse Acute and Right": "Angles - Obtuse, Acute, and Right",
    "Parts of a Polynomial Terms Coefficient Monomial Exponent Variable": "Parts of a Polyomial, Terms, Coefficient, Monomial, Exponent, Variable"
}

# Transform skill_id to skill-indices, ie. values from 1 to max
data_parts = pd.DataFrame(columns=chosen_columns)
skill_occurences_threshold = 30
printProgressBar(0, num_skills, prefix='Progress:',
                 suffix='Complete', length=50)
for i in range(num_skills):
    skill_name = skill_names[i, 0]
    filtered = data[data["skill_name"] == skill_name]
    if (skill_name in skill_names_to_change):
        filtered = data[data["skill_name"] ==
                        skill_names_to_change[skill_name]]
    filtered.loc[:, "skill_id"] = i + 1

    # threshold of skill occurences
    # if len(filtered) < skill_occurences_threshold:
    #     unused_skills.append(skill_name)
    #     continue

    data_parts = pd.concat([data_parts, filtered])
    printProgressBar(i+1, num_skills, prefix='Progress for skill_ids:',
                     suffix='Complete', length=50)

data = data_parts.sort_values(by=["user_id"])

grouped_data = data.groupby("user_id")
unique_users = data["user_id"].unique()

unique_skills_num = len(data["skill_id"].unique())
unique_users_num = len(data["user_id"].unique())
print('Number of skills: {}, number of students: {}'.format(
    unique_skills_num, unique_users_num))
print("Unused Skills: {}".format(unused_skills))


# In[4]

def four_to_three(filename, data):
    with open(filename, 'w') as fh:
        for student in data.user_id.unique():
            student_data = data[data['user_id'] ==
                                student][['skill_id', 'correct']].values
            num_responses = student_data.shape[0]
            # Print number of responses
            print('{:d}'.format(num_responses), file=fh)
            # Print skill_ids
            for i in range(num_responses-1):
                print('{:d}'.format(student_data[i, 0]), end=',', file=fh)
            # last line
            print('{:d}'.format(student_data[num_responses-1, 0]), file=fh)
            # Print correct value (0/1)
            for i in range(num_responses-1):
                print('{:d}'.format(student_data[i, 1]), end=',', file=fh)
            # last line
            print('{:d}'.format(student_data[num_responses-1, 1]), file=fh)

# In[5]:
# Save `data` to csv


data.to_csv(path.join(rootdir, "data/"+data_set+"/"+data_set+".csv"),
            index=None)

# In[51]:
# Create custom splits (train-test only)
data_test = pd.DataFrame(columns=chosen_columns)
data_train = pd.DataFrame(columns=chosen_columns)
# this is here due to drops happening below
training_data_entries_num = 2 * math.ceil(len(data)/3)
train_splits = {
    0: pd.DataFrame(columns=chosen_columns),
    1: pd.DataFrame(columns=chosen_columns),
    2: pd.DataFrame(columns=chosen_columns),
    3: pd.DataFrame(columns=chosen_columns),
    4: pd.DataFrame(columns=chosen_columns)
}
validation_splits = {
    0: pd.DataFrame(columns=chosen_columns),
    1: pd.DataFrame(columns=chosen_columns),
    2: pd.DataFrame(columns=chosen_columns),
    3: pd.DataFrame(columns=chosen_columns),
    4: pd.DataFrame(columns=chosen_columns)
}

# train initialization for skill coverage
printProgressBar(0, unique_skills_num, prefix='Progress:',
                 suffix='Complete', length=50)
for index, entry in data.iterrows():
    if len(data_train["skill_id"]) == unique_skills_num:
        break
    entry_user = entry["user_id"]
    entry_skill = entry["skill_id"]
    if entry_skill in np.int64(data_train["skill_id"]):
        continue
    data_train = data_train.append(entry, ignore_index=True)
    data.drop(index, inplace=True)
    printProgressBar(len(data_train["skill_id"]), unique_skills_num, prefix='Initializing training set:',
                     suffix='Complete', length=50)

# populate train set with the users already in it
unique_users_train = data_train["user_id"].unique()
for user in unique_users_train:
    user_entries = data[data["user_id"] == user]
    user_entries_index = user_entries.index
    data_train = data_train.append(user_entries, ignore_index=True)
    data.drop(user_entries_index, inplace=True)

# test initialization for skill coverage
printProgressBar(0, unique_skills_num, prefix='Progress:',
                 suffix='Complete', length=50)
for index, entry in data.iterrows():
    if len(data_test["skill_id"]) == unique_skills_num:
        break
    entry_user = entry["user_id"]
    entry_skill = entry["skill_id"]
    if entry_skill in np.int64(data_test["skill_id"]) or entry_user in np.int64(data_train["user_id"]):
        continue
    data_test = data_test.append(entry, ignore_index=True)
    data.drop(index, inplace=True)
    printProgressBar(len(data_test["skill_id"]), unique_skills_num, prefix='Initializing test set:',
                     suffix='Complete', length=50)

# populate test set with the users already in it
unique_users_test = data_test["user_id"].unique()
for user in unique_users_test:
    user_entries = data[data["user_id"] == user]
    user_entries_index = user_entries.index
    data_test = data_test.append(user_entries, ignore_index=True)
    data.drop(user_entries_index, inplace=True)

# fill training set up to ~70% of data
user_groups = data.groupby("user_id")
printProgressBar(0, training_data_entries_num, prefix='Progress:',
                 suffix='Complete', length=50)
for name, group in user_groups:
    if len(data_train) >= training_data_entries_num:
        break
    data_train = data_train.append(group, ignore_index=True)
    group_entries = data[data["user_id"] == name]
    group_entries_index = group_entries.index
    data.drop(group_entries_index, inplace=True)
    printProgressBar(len(data_train), training_data_entries_num, prefix='Filling training set:',
                     suffix='Complete', length=50)

# fill test set with the remaining data
print("\n Filling the test set...")
data_test = data_test.append(data, ignore_index=True)

# populate train - validation splits
user_groups_list = [group for _, group in data_train.groupby("user_id")]
max_validation_entries = 0.2 * len(data_train)
for i in range(5):
    print("Populating train / validation split {}...".format(i+1))
    random.shuffle(user_groups_list)

    validation_entries_index = []
    for group in user_groups_list:
        if len(validation_splits[i]) >= max_validation_entries:
            break
        validation_splits[i] = validation_splits[i].append(
            group, ignore_index=True)
        validation_entries_index = np.append(
            validation_entries_index, group.index)

    train_splits[i] = data_train.drop(validation_entries_index)


for i in range(5):
    iter = str(i + 1)
    print("Writting train/validation split {} to csv...".format(iter))

    data_train_split = train_splits[i]
    data_validation_split = validation_splits[i]

    # ### Save train data in 4-column format
    filename = path.join(
        rootdir, "data/"+data_set+"/new_splits/"+data_set+"_train"+iter+"_4columns.csv")
    data_train_split.to_csv(filename, index=None)
    filename = path.join(
        rootdir, "data/"+data_set+"/new_splits/"+data_set+"_valid"+iter+"_4columns.csv")
    data_validation_split.to_csv(filename, index=None)

    # ### Save train data in 3-lines format
    filename = path.join(
        rootdir, "data/"+data_set+"/new_splits/"+data_set+"_train"+iter+"_3lines.csv")
    four_to_three(filename, data_train_split)
    filename = path.join(
        rootdir, "data/"+data_set+"/new_splits/"+data_set+"_valid"+iter+"_3lines.csv")
    four_to_three(filename, data_validation_split)

filename = path.join(
    rootdir, "data/"+data_set+"/new_splits/"+data_set+"_train_4columns.csv")
data_train.to_csv(filename, index=None)
filename = path.join(
    rootdir, "data/"+data_set+"/new_splits/"+data_set+"_test_4columns.csv")
data_test.to_csv(filename, index=None)

filename = path.join(
    rootdir, "data/"+data_set+"/new_splits/"+data_set+"_train.csv")
four_to_three(filename, data_train)
filename = path.join(
    rootdir, "data/"+data_set+"/new_splits/"+data_set+"_test.csv")
four_to_three(filename, data_test)

print("\n Done!")
