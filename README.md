# Lost in Translationese? Reducing Translation Effect Using Abstract Meaning Representation

This is the code used for the AMR parse-then-generate technique and approach evaluation in the EACL 2024 paper called "Lost in Translationese? Reducing Translation Effect Using Abstract Meaning Representation" by Shira Wein and Nathan Schneider.

In order to run the parse-then-gen.py file, you will need to have amrlib installed and parsing and generation models downloaded. You can follow the amrlib installation instructions here: https://amrlib.readthedocs.io/en/latest/install/
Note that some code for this file was adapted from the following repository: https://github.com/Heidelberg-NLP/simple-xamr

parse-then-gen.py is run via the following command, where -l indicates language and -f indicates file containing the sentences you intend to parse-then-generate from:
```
python p-then-g.py -l ['source-language'] -f [sentences.txt]
```

The paraphrase.ipynb notebook contains code for paraphrase generation and evaluation, and the udavg.ipynb notebook also contains code for evaluation.
