import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.model_selection import train_test_split

"""
To generate more discrimination: ~5k 
Method:
    Replace words
    Adjoin discri with normal data
    Adjoin discri with discri data
"""

# load data
train_data = pd.read_csv('../data/train.csv', encoding="utf-8") 
test_data = pd.read_csv('../data/test.csv') 


# We have 2242 discri records
discrimination_data = train_data[train_data.label == 1]
normal_data = train_data[train_data.label == 0]

# Empty pd dataframe
generate_data =pd.DataFrame(columns=["id","label","tweet"])
# Id for generated data
idx = 1

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def replace_word(count):
    replacement = {"people":"human being", 
                    "fun": "interesting",
                    "happy":"cheer", 
                    "love": "like",
                    "cute": "cat",
                    "positive": "friendly",
                    "libtard":"liberal",
                    "white":"black",
                    "black":"white",
                    "healthy":"fit",
                    }
    loop_idx = 0
    while count > 0 and loop_idx < len(discrimination_data):
        original_string = discrimination_data["tweet"].values[loop_idx]
        generate_string = replace_all(original_string, replacement)
        loop_idx+=1
        if original_string != generate_string:
            generate_data.loc[-1] = [idx, 1, generate_string]
            generate_data.index = generate_data.index + 1
            count-=1


def adjoin_dis_normal(count):
    adjoin_dis_data = discrimination_data.sample(n=count)
    adjoin_normal_data = normal_data.sample(n=count)
    
    for idx in range(count):
        normal_string = adjoin_normal_data["tweet"].values[idx]
        discri_string = discrimination_data["tweet"].values[idx]
        insert_idx = random.randint(0, len(normal_string))
        
        generate_string = normal_string[:insert_idx] + " " + discri_string + " " + normal_string[insert_idx:]

        generate_data.loc[-1] = [idx, 1, generate_string]
        generate_data.index = generate_data.index + 1
    

    

def adjoin_dis_dis(count):
    adjoin_dis_data1 = discrimination_data.sample(n=count)
    adjoin_dis_data2 = discrimination_data.sample(n=count)
    
    for idx in range(count):
        discri_string1 = adjoin_dis_data1["tweet"].values[idx]
        discri_string2 = adjoin_dis_data2["tweet"].values[idx]
        insert_idx = random.randint(0, len(discri_string1))
        
        generate_string = discri_string1[:insert_idx] + " " + discri_string2 + " " + discri_string1[insert_idx:]
        generate_data.loc[-1] = [idx, 1, generate_string]
        generate_data.index = generate_data.index + 1


# Generate new discrimination data
replace_word(2242)
adjoin_dis_dis(1000)
adjoin_dis_dis(600)


"""
Spite data into test and train; the rate of test and train is 3:7
The rate of discrimination and normal data is 1:1.2
"""
discrimination_data.append(generate_data)
normal_data_count = int(len(discrimination_data) * 1.2)
normal_data = normal_data.sample(normal_data_count)
new_normal_train_data,new_normal_test_data = train_test_split(normal_data, test_size=0.3)
new_discri_train_data,new_discri_test_data = train_test_split(discrimination_data, test_size=0.3)
new_train_data = new_normal_train_data.append(new_discri_train_data)
new_test_data = new_normal_test_data.append(new_discri_test_data)
new_train_data.to_csv("./train.csv",sep='\t')
new_test_data.to_csv("./test.csv",sep='\t')

