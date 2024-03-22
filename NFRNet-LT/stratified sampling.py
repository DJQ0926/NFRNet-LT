import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("E:\Lesson5\processed_Promise.csv")


class_counts = data['LABEL'].value_counts()

train_data = pd.DataFrame(columns=data.columns)
test_data = pd.DataFrame(columns=data.columns)

for label, count in class_counts.items():

    train_count = int(count * 0.7)  
    test_count = count - train_count  
    

    class_data = data[data['LABEL'] == label]
    train_class_data, test_class_data = train_test_split(class_data, train_size=train_count, test_size=test_count, random_state=42)
    
 
    train_data = train_data.append(train_class_data)
    test_data = test_data.append(test_class_data)


train_data.to_csv("save_train_path", index=False)
test_data.to_csv("save_test_path", index=False)
