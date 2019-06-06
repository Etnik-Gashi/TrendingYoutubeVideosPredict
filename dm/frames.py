import tkinter as tk
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import csv
import ipython_genutils as ip
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import datetime
import wordcloud
import program as p1
import program2 as p2


PLOT_COLORS = ["#268bd2", "#0052CC", "#FF5722", "#b58900", "#003f5c"]
pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('font', family='Arial', weight='400', size=10)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)



#process Decision tree
df = pd.read_csv(r"./TrendingJoTrending.csv", header=None)
df[0] = pd.to_numeric(df[0], errors='coerce')
df = df.replace(np.nan, 0, regex=True)

df[1] = pd.to_numeric(df[1], errors='coerce')
df = df.replace(np.nan, 1, regex=True)

df[2] = pd.to_numeric(df[2], errors='coerce')
df = df.replace(np.nan, 2, regex=True)

df[3] = pd.to_numeric(df[3], errors='coerce')
df = df.replace(np.nan, 3, regex=True)

df[4] = pd.to_numeric(df[4], errors='coerce')
df = df.replace(np.nan, 4, regex=True)

df[5] = pd.to_numeric(df[5], errors='coerce')
df = df.replace(np.nan, 5, regex=True)

feature_cols = [0,1,2,3,4,5]
X = df[feature_cols] # Features
y = df[6] # Target variable


class Windows(tk.Tk):
    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)
        container=tk.Frame(self)
        container.grid()

        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)

        self.frames={}

        for F in (StartPage,PageOne,PageTwo,PageThree):
                
            frame=F(container,self)

            self.frames[F]=frame

            frame.grid(row=0,column=0,sticky="nsew")
        
        self.show_frame(StartPage)

    def show_frame(self,cont):
        frame=self.frames[cont]
        frame.tkraise()





class StartPage(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        #self.title("Trending Youtube Videos Statistics")
        #self.geometry('500x400')
        #self.config(bg = "lightblue")
        lbl=tk.Label(self,text='Welcome to TYVS',fg='red')
        lbl.place(relx=.5, rely=.4, anchor="center")

        btn=tk.Button(self,text='Start',command=lambda:controller.show_frame(PageOne),fg='blue')
        btn.place(relx=.5, rely=.5, anchor="center")

class PageOne(tk.Frame):
       

    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        
        tk.Label(self, text="Te dhenat e videos").grid(row=1,column=2)
        tk.Label(self, text="Data e publikuar").grid(row=2,column=1)
        tk.Label(self, text="Cat ID").grid(row=3,column=1)
        tk.Label(self, text="Views").grid(row=4,column=1)
        tk.Label(self, text="Likes").grid(row=5,column=1)
        tk.Label(self, text="Dilikes").grid(row=6,column=1)
        tk.Label(self, text="Comments").grid(row=7,column=1)
        
        e1 = tk.Entry(self) 
        e1.grid(row=2, column=2) 
        e2 = tk.Entry(self) 
        e2.grid(row=3, column=2) 
        e3 = tk.Entry(self) 
        e3.grid(row=4, column=2) 
        e4 = tk.Entry(self) 
        e4.grid(row=5, column=2) 
        e5 = tk.Entry(self) 
        e5.grid(row=6, column=2) 
        e6 = tk.Entry(self) 
        e6.grid(row=7, column=2) 
       
        
        lbl5=tk.Label(self,text="")
        lbl5.grid()
        def insert():
            fields=[e1.get(),e2.get(),e3.get(),e4.get(),e5.get(),e6.get()]
            with open(r'./TrendingJoTrending.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
      
        def predict():
            df = pd.read_csv(r"./TrendingJoTrending.csv", header=None)
            df[0] = pd.to_numeric(df[0], errors='coerce')
            df = df.replace(np.nan, 0, regex=True)

            df[1] = pd.to_numeric(df[1], errors='coerce')
            df = df.replace(np.nan, 1, regex=True)

            df[2] = pd.to_numeric(df[2], errors='coerce')
            df = df.replace(np.nan, 2, regex=True)

            df[3] = pd.to_numeric(df[3], errors='coerce')
            df = df.replace(np.nan, 3, regex=True)

            df[4] = pd.to_numeric(df[4], errors='coerce')
            df = df.replace(np.nan, 4, regex=True)

            df[5] = pd.to_numeric(df[5], errors='coerce')
            df = df.replace(np.nan, 5, regex=True)
            feature_cols = [0,1,2,3,4,5]
            X = df[feature_cols] # Features
            y = df[6]
            test_row=(df.shape[0])-1 #rreshti qe predikon
            train_idx=np.arange(X.shape[0])!=test_row
            test_idx=np.arange(X.shape[0])==test_row
            print(df.shape[0])


            X_train=X[train_idx]
            y_train=y[train_idx]

            X_test=X[test_idx]
            y_test=y[test_idx]
            #Create Decision Tree classifer object
            clf = DecisionTreeClassifier(max_depth=5)
            #Train Decision Tree Classifer
            clf = clf.fit(X_train,y_train)
            #Predict the response for test dataset
            y_pred = clf.predict(X_test)
            lbl2.config(text=y_pred)#,)
                        
                
        inBtn = tk.Button(self,text="Insert",command=insert,fg='blue')
        inBtn.grid(row=8,column=1)
        predBtn = tk.Button(self,text="Predict",command=predict,fg='blue')
        predBtn.grid(row=8,column=2)
        deskBtn = tk.Button(self,text="Descript",command=lambda:controller.show_frame(PageTwo),fg='blue')
        deskBtn.grid(row=8,column=3)
        nextBtn = tk.Button(self,text="Parashiko Nr Rreshti",command=lambda:controller.show_frame(PageThree),fg='blue')
        nextBtn.grid(row=8,column=4)
        lblOut=tk.Label(self,text="Rezultati:")
        lblOut.grid(row=9,column=1)
        lbl2=tk.Label(self,text="")
        lbl2.grid(row=9,column=2)
        lbl3=tk.Label(self,text="")
        lbl3.grid(row=9,column=3)
        lbl4=tk.Label(self,text="")
        lbl4.grid(row=9,column=4)
        

class PageTwo(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        
        backBtn = tk.Button(self,text="Back",command=lambda:controller.show_frame(PageOne),fg='blue')
        backBtn.grid()#row=3,column=4)
        exitBtn = tk.Button(self,text="Exit",command=lambda:controller.show_frame(StartPage),fg='blue')
        exitBtn.grid()
        lbl=tk.Label(self,text="Vetite e Datasetit")
        lbl.grid()
        text = tk.Text(self)
        text.config(width=70,height=10)
        text.insert(tk.END, df.describe())
        text.grid()

        lbl=tk.Label(self,text="Lidhshmeria mes elementeve")
        lbl.grid()
        text2 = tk.Text(self)
        text2.config(width=50,height=7)
        text2.insert(tk.END, df.corr())
        text2.grid()
        lbl=tk.Label(self,text="Korrelacionet")
        lbl.grid()
        corrBtn = tk.Button(self,text="Shfaq",command=lambda:p1.show(),fg='blue')
        corrBtn.grid()#row=3,column=4)
        


class PageThree(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        backBtn = tk.Button(self,text="Back",command=lambda:controller.show_frame(PageTwo),fg='blue')
        backBtn.grid()#row=3,column=4)
        exitBtn = tk.Button(self,text="Exit",command=lambda:controller.show_frame(StartPage),fg='blue')
        exitBtn.grid()
        lbl=tk.Label(self,text="Parashiko ne baze te rreshtit ne dataset:")
        lbl.grid()
        lbl1=tk.Label(self,text="")
        lbl1.grid()
        e = tk.Entry(self) 
        e.grid()
        lblA=tk.Label(self,text="")
        lblA.grid()
        lblM=tk.Label(self,text="")
        lblM.grid()
        def pred():
            
            test_row=float(e.get()) #rreshti qe predikon
            train_idx=np.arange(X.shape[0])!=test_row
            test_idx=np.arange(X.shape[0])==test_row
            print(df.shape[0])

            X_train=X[train_idx]
            y_train=y[train_idx]

            X_test=X[test_idx]
            y_test=y[test_idx]


            #Create Decision Tree classifer object
            clf = DecisionTreeClassifier(max_depth=5)


            #Train Decision Tree Classifer
            clf = clf.fit(X_train,y_train)

            #Predict the response for test dataset
            y_pred = clf.predict(X_test)
            lblA.config(text=metrics.accuracy_score(y_test, y_pred))
            if y_pred[0]==np.asarray(y_test).reshape(-1)[0]:
                lblM.config(text="Ka parashikuar mirë")
            elif y_pred[0]!=np.asarray(y_test).reshape(-1)[0]:
                lblM.config(text="Nuk ka parashikuar mirë")
        
        predBtn = tk.Button(self,text="Prediko",command=pred,fg='blue')
        predBtn.grid()
        
        



        
        
                
        
    
app=Windows()
app.mainloop()
