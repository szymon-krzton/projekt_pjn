import pandas as pd

def get_lang_name(lang_shortcut):
    url = "https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes"
    df = pd.read_html(url, match="ISO language name")
    langs = dict(zip(df[0]['639-1'], df[0]['ISO language name']))
    return langs[lang_shortcut]