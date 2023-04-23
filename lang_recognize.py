import fasttext
import short_to_long as stl

class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1) # returns top 2 matching languages
        return predictions

if __name__ == '__main__':
    LANGUAGE = LanguageIdentification()
    lang = LANGUAGE.predict_lang("Dzień dobry")
    print(f'Language of given text is {stl.get_lang_name(lang[0][0][-2:])} with probability of {int(round(lang[1][0], 2)*100)}%')