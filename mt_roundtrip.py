# Some code adapted from https://github.com/Heidelberg-NLP/simple-xamr

from nmt_english import Translator
import multiprocessing
import os
import locale
import sys

def truncate_files(filename):
    # cut off the last 5 lines to omit the obsolete AMR graph
    with open(filename, mode='r+', encoding='utf-8') as f:
        lines = f.readlines()[:-5]
        print(lines)
        f.truncate()
        f.seek(0)
        f.writelines(lines)

if __name__ == '__main__':
    arguments = sys.argv
    
    source_language = arguments[2]
    file_to_translate = arguments[4]
        
    print("\nParsing file", file_to_translate, "from", source_language + ".\n")
    
    multiprocessing.freeze_support()
    
    # Translate file and save it to translations folder
    translator = Translator()
    translator.load_sentences(file_to_translate)
    translator.translate(source_language=source_language, target_language='fr')
    translator.save_translation("translated_nmt.txt")
    translation_file = "translated_nmt.txt"

    translator.load_sentences("translated_nmt.txt")
    translator.translate(source_language='fr', target_language='en')
    translator.save_translation("back_trans.txt")
    translation_file = "back_trans.txt"
