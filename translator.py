from googletrans import Translator
from nepali_unicode_converter.convert import Converter

converter = Converter() 
translator = Translator()

def translate(text):
    nepaliText = translator.translate(text,src= 'en' ,dest="ne")
    return nepaliText.text
