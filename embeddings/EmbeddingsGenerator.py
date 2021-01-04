# %%
import sys
import numpy as np
import pandas as pd

# %%
# path needs to change depending on user
path = 'C:/Users/Vasilis Nikiforidis/Desktop/Thesis/MyCode/'
sys.path.append(path)
pre_trained = path + "embeddings/enwiki_20180420_100d.txt"
data_set = "assistment_2009_corrected"
data_format = "3lines"
data_set_folder = "data/" + data_set + "/"
dataset_file = path + data_set_folder + "assistment_2009_corrected.csv"

# %%


def read_data():
    data = pd.read_csv(dataset_file, delimiter=',', header=0, encoding='ANSI')
    skills = np.array(data['skill_name'].unique())
    skill_ids = np.array(data['skill_id'].unique()).astype(int)
    students = np.array(data['user_id'].unique())
    data = data.astype({"skill_id": int})
    return data, skills, skill_ids, students


def convert(lst):
    return (lst.split())


# %%
dataset, skills, skill_ids, students = read_data()

# generate skill embeddings

vec_arr = []
word_arr = []
not_found = []
in_skill_name = []
s = skills.shape[0]
d = 300
i = 0
counter = 0
sentence_emb = np.zeros([s, d])
sentence_emb[0, :] = 0.0
for i in range(s):
    low_words = skills[i].lower()
    skills_low = convert(low_words)
    vectors = np.zeros
    print('Skill name:{} - {}'.format(i, skills[i]))
    for word in skills_low:
        # the format of pre_trained file is : word number1 number2 number3 .........number300 - one line for each word, tab delimited
        with open(pre_trained, 'r') as f:
            while True:
                counter += 1
                try:
                    line = f.readline()
                    if not line:
                        # finding the word that doesn't exist in pre-trained file
                        not_found.append(word)
                        # finding the skill name in which one or more words have not been found
                        in_skill_name.append(skills[i])
                        break
                    if counter % 10000000 == 0:
                        print('line = {}'.format(line))
                    if word + " " == line[:len(word)+1]:
                        str_list = line[len(word)+1:].split(" ")
                        vec = [float(num) for num in str_list]
                        break
                except Exception as e:
                    e = None
            vec_arr.append(vec)
            word_arr.append(word)
            print("{}".format(word))
    vectors = np.array(vec_arr)
    vec_arr.clear()
    words = np.array(word_arr)
    not_found_words = np.array(not_found)
    no_word_in_skill = np.array(in_skill_name)
    # skill name embedding vector is the sum of the words embeddings of the skill name
    sentence_emb[i, :] = np.sum(vectors, axis=0)
f.close()

# %%
# dataset.to_csv(path+"assistment2009_corrected.csv",sep=",", index=False)
pd.DataFrame(sentence_emb).to_csv(
    path+"embeddings" + data_set + "/skill_name_embeddings_corrected_100.csv", sep=",", index=False, header=None)
pd.DataFrame(skills).to_csv(
    path+"embeddings" + data_set + "/skill_names_corrected.csv", sep=",", index=False, header=None)
pd.DataFrame(not_found_words).to_csv(
    path+"embeddings" + data_set + "/not_found_words_corrected.csv", sep=",", index=False, header=None)
pd.DataFrame(no_word_in_skill).to_csv(
    path+"embeddings" + data_set + "/no_words_in_skill_corrected.csv", sep=",", index=False, header=None)
