# Some code adapted from https://github.com/Heidelberg-NLP/simple-xamr

import amrlib
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
        
    if not os.path.exists('translations'):
        os.makedirs('translations')
        
    if not os.path.exists('AMRgraphs'):
        os.makedirs('AMRgraphs')
        
    print("\nParsing file", file_to_translate, "from", source_language + ".\n")
    
    multiprocessing.freeze_support()
    
    translations = sorted(os.listdir("translations"))  # where the translated files are stored
    amr_graphs = sorted(os.listdir("AMRgraphs"))  # where to store the parsed AMR graphs

    my_file = open(file_to_translate, "r")
    data = my_file.read()
    sent_list = data.split("\n")
    print("sent list: ", sent_list)

    stog = amrlib.load_stog_model()
    print("Parse Model loaded.")
    graphs = stog.parse_sents(sent_list)
    print("graphs ", graphs)

    with open("graphs.txt", 'w') as gp:
        for item in graphs:
            # write each item on a new line
            gp.write("%s\n" % item)
    gp.close()

    gtos = amrlib.load_gtos_model()
    print("Gen Model loaded.")
    sents, _ = gtos.generate(graphs)
    print("sents ", sents)

    file_path = file_to_translate[:-4] + ".back_sents.txt"
    with open(file_path, 'w') as fp:
        for item in sents:
            # write each item on a new line
            fp.write("%s\n" % item)

    print("Sentences stored to file.")
    fp.close()
