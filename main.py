import tkinter as tk
import pandas as pd
from tkinter import messagebox, ttk
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from lang_recognize import LanguageIdentification
import short_to_long as stl


def search(user):
    reddit = praw.Reddit(client_id='EuotFvMTAV1Sdn8pvS-yog', client_secret='v6qKmbKFIzlQd-amcpYej4B7sWvE6Q',
                         username='xDriss', password='', user_agent='simple sentiment analysis')

    sentence = []

    try:
        for post in reddit.redditor(user).comments.top(time_filter='all'):
            sentence.append(post.body)
    except:
        return False
    else:
        return sentence


def sentiment_analysis(posts):
    scores = []
    language = []

    # sa = SentimentIntensityAnalyzer()
    lr = LanguageIdentification()

    for post in posts:
        # scores.append(sa.polarity_scores(post)['compound'])
        lang = lr.predict_lang(post)
        scores.append(f'{int(round(lang[1][0], 2)*100)}%')
        language.append(stl.get_lang_name(lang[0][0][-2:]))

    return pd.DataFrame(list(zip(posts, language, scores)), columns=['Comment', 'Language', 'Score'])


class GUI:
    def __init__(self, master):
        self.master = master
        master.title("GUI Interface")

        self.label = tk.Label(master, text="Enter Text:")
        self.label.pack()

        self.entry = tk.Entry(master)
        self.entry.pack()

        self.button = tk.Button(
            master, text="Sentiment analysis", command=self.display)
        self.button.pack()

    def display(self):
        user = self.entry.get()
        if search(user) is not False:
            data = search(user)
            df = sentiment_analysis(data)
            self.display_dataframe(df)
        else:
            messagebox.showerror(
                "Error", "Wrong username or user has no posts.")

    def display_dataframe(self, df):
        top = tk.Toplevel()
        top.title("Dataframe Display")

        treeview = ttk.Treeview(top)
        treeview.pack(fill=tk.BOTH, expand=True)

        treeview["columns"] = list(df.columns)
        treeview["show"] = "headings"

        for column in treeview["columns"]:
            treeview.heading(column, text=column)

        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            treeview.insert("", "end", values=row)


if __name__ == '__main__':
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
