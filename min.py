from tkinter import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tkinter import messagebox

# Initialize the main window
main = tk.Tk()
main.title("Machine Learning-Based Crop Recommendation System")
main.geometry("1000x650")

# Global variables
filename = None
dataset = None
X = None
Y = None
x_train = None
y_train = None
x_test = None
y_test = None
knn = None
rf = None
accuracy = []
precision = []
recall = []
fscore = []

# Function to upload dataset
def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    if filename:
        text.insert(END, filename + ' Loaded\n')
        dataset = pd.read_csv(filename)
        text.insert(END, str(dataset.head()) + "\n\n")
    else:
        messagebox.showerror("Error", "Please upload a dataset file.")

# Function to preprocess dataset
def preprocessDataset():
    global X, Y, le, dataset, x_train, y_train, x_test, y_test
    if dataset is None:
        messagebox.showerror("Error", "Dataset not loaded. Please upload a dataset first.")
        return
    le = LabelEncoder()
    text.delete('1.0', END)
    dataset.fillna(0, inplace=True)
    text.insert(END, str(dataset.head()) + "\n\n")

    X = dataset.drop(['label'], axis=1).values
    Y = dataset['label'].values
    Y = le.fit_transform(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END, "Dataset Length: " + str(len(X)) + "\n")
    text.insert(END, "Training Set Size: " + str(len(x_train)) + "\n")
    text.insert(END, "Test Set Size: " + str(len(x_test)) + "\n")

# Function to display the dataset distribution graph
def showDataDistribution():
    global dataset
    if dataset is None:
        messagebox.showerror("Error", "Dataset not loaded. Please upload a dataset first.")
        return
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', hue='label', data=dataset, palette="Set3", legend=False)
    plt.xlabel("Crop Categories")
    plt.ylabel("Count")
    plt.title("Dataset Distribution")
    plt.show()

# Function to apply KNeighborsClassifier
def custom_knn_classifier():
    global knn
    if x_train is None or y_train is None:
        messagebox.showerror("Error", "Please preprocess the dataset first.")
        return
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    acc = accuracy_score(y_test, prediction)
    prec = precision_score(y_test, prediction, average='macro')
    rec = recall_score(y_test, prediction, average='macro')
    f1 = f1_score(y_test, prediction, average='macro')
    text.insert(END, f"KNeighborsClassifier Accuracy: {acc}\n")
    text.insert(END, f"Precision: {prec}\n")
    text.insert(END, f"Recall: {rec}\n")
    text.insert(END, f"F1 Score: {f1}\n")
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    fscore.append(f1)

# Function to apply RandomForestClassifier
def Randomforestclassifier():
    global rf
    if x_train is None or y_train is None:
        messagebox.showerror("Error", "Please preprocess the dataset first.")
        return
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    prediction = rf.predict(x_test)
    acc = accuracy_score(y_test, prediction)
    prec = precision_score(y_test, prediction, average='macro')
    rec = recall_score(y_test, prediction, average='macro')
    f1 = f1_score(y_test, prediction, average='macro')
    text.insert(END, f"RandomForestClassifier Accuracy: {acc}\n")
    text.insert(END, f"Precision: {prec}\n")
    text.insert(END, f"Recall: {rec}\n")
    text.insert(END, f"F1 Score: {f1}\n")
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    fscore.append(f1)

# Function to predict new data
def predict():
    global knn, rf, le
    if knn is None or rf is None:
        messagebox.showerror("Error", "Please train the classifiers first.")
        return
    user_input = user_input_entry.get()
    if not user_input:
        messagebox.showerror("Error", "Please enter input values.")
        return
    try:
        input_values = np.array(user_input.split(',')).astype(float).reshape(1, -1)
        knn_pred = le.inverse_transform(knn.predict(input_values))
        rf_pred = le.inverse_transform(rf.predict(input_values))
        text.insert(END, f"KNN Prediction: {knn_pred[0]}\n")
        text.insert(END, f"Random Forest Prediction: {rf_pred[0]}\n")
    except ValueError:
        messagebox.showerror("Error", "Invalid input format. Please enter comma-separated numerical values.")

# Function to show performance comparison graph
def graph():
    if not accuracy or not precision or not recall or not fscore:
        messagebox.showerror("Error", "Please run the classifiers first.")
        return
    df = pd.DataFrame({
        'Algorithms': ['KNN', 'Random Forest'],
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': fscore
    })

    df = df.melt(id_vars='Algorithms', var_name='Parameters', value_name='Value')
    pivot_df = df.pivot_table(index='Parameters', columns='Algorithms', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar')
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Function to close the application
def close():
    main.destroy()

# GUI layout
font = ('times', 16, 'bold')
title = Label(main, text='Machine Learning-Based Crop Recommendation System', justify=LEFT)
title.config(bg='lavender blush', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=200, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=500, y=100)
preprocessButton.config(font=font1)

knnButton = Button(main, text="KNeighborsClassifier", command=custom_knn_classifier)
knnButton.place(x=500, y=150)
knnButton.config(font=font1)

RFButton = Button(main, text="RandomForestClassifier", command=Randomforestclassifier)
RFButton.place(x=200, y=150)
RFButton.config(font=font1)

predictLabel = Label(main, text="Enter Input (comma-separated):", font=font1)
predictLabel.place(x=200, y=200)
user_input_entry = Entry(main, width=40)
user_input_entry.place(x=500, y=200)
predictButton = Button(main, text="Predict", command=predict)
predictButton.place(x=500, y=250)
predictButton.config(font=font1)

dataGraphButton = Button(main, text="Show Data Distribution", command=showDataDistribution)
dataGraphButton.place(x=200, y=250)
dataGraphButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=500, y=300)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=200, y=300)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=120)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=350)
text.config(font=font1)
main.config(bg='LightSteelBlue1')
main.mainloop()
