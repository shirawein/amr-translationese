!pip install transformers

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

import spacy
import re
# import nltk
# from nltk import bigrams
# from nltk import ngrams

nlp = spacy.load("en_core_web_sm")
reader = open('1000translations.txt', 'r')

# Paraphrase Generation
model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')

#using model from https://huggingface.co/eugenesiow/bart-paraphrase?text=Madam+President+%2C+coinciding+with+this+year+%27+s+first+part-session+of+the+European+Parliament+%2C+a+date+has+been+set+%2C+unfortunately+for+next+Thursday+%2C+in+Texas+in+America+%2C+for+the+execution+of+a+young+34+year-old+man+who+has+been+sentenced+to+death+.+We+shall+call+him+Mr+Hicks+.

para_bart_large = open('para_bart.txt', 'w')

while True:
  input_sentence = reader.readline()
  if not input_sentence:
    break
  batch = tokenizer(input_sentence, return_tensors='pt')
  generated_ids = model.generate(batch['input_ids'])
  generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
  ret_text = str(generated_sentence)[2:-2]
  print(ret_text)
  para_bart_large.write(ret_text)
  para_bart_large.write('\n')

para_bart_large.close()

reader = open('1000translations.txt', 'r')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

#using model from https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base?text=Madam+President+%2C+coinciding+with+this+year+%27+s+first+part-session+of+the+European+Parliament+%2C+a+date+has+been+set+%2C+unfortunately+for+next+Thursday+%2C+in+Texas+in+America+%2C+for+the+execution+of+a+young+34+year-old+man+who+has+been+sentenced+to+death+.+We+shall+call+him+Mr+Hicks+.

para_t5 = open('para_t5.txt', 'w')

while True:
  input_sentence = reader.readline()
  if not input_sentence:
    break
  output = str(paraphrase(input_sentence))[2:-2]
  print(output)
  para_t5.write(output)
  para_t5.write('\n')

para_t5.close()

# Evaluation

para_bart_reader = open('para_bart.txt', 'r')

para_t5_reader = open('para_t5.txt', 'r')

uni_cohesive_markers = ["because", "besides", "but", "consequently", "despite", "except", "further", "furthermore", "hence", "however", "instead", "maybe", "moreover", "nevertheless", "otherwise",  "since", "so", "therefore",  "though", "thus",  "yet", "concerning"]
bi_cohesive_markers = [["as","for"], ["as","to"], ["even", "if"], ["even","though"], ["in", "addition"], ["in","conclusion"], ["in","spite"], ["referring","to"], ["the", "former"], ["the", "latter"], ["this", "implies"]]
tri_cohesive_markers = [["in","other","words"], ["is","to","say"], ["on","account","of"], ["on","the","contrary"], ["with","reference","to"], ["with","regard","to"]]
quat_cohesive_markers = ["on","the","other","hand"]

def dist_from_root(tok, count):
  #print("tok ", tok)
  if tok.dep_ == 'ROOT':
    #print("count ", count)
    return count
  else:
    return dist_from_root(tok.head, count+1)

def summed_dist_for_sent_length(sum_dist, sent_length):
  if sent_length not in dist_dict:
    dist_dict[sent_length] = []
  dist_dict[sent_length].append(sum_dist)

def summed_dist(sent):
  doc = nlp(sent)
  total_count = 0

  for token in doc:
    total_count += dist_from_root(token,0)
    #print(total_count)

  sent_len = len(sent.split())

  summed_dist_for_sent_length(total_count,sent_len)

def collect_all_unique_tokens(sent):
  doc = nlp(sent)
  for token in doc:
    list_all_tokens.append(token)
    list_pos_tags.append(token.pos_)
    if token.lemma_ not in list_unique_tokens:
      list_unique_tokens.append(token.lemma_)

def individual_sent_tokens(sent):
  temp_tokens = []
  uniq_tokens = []
  doc = nlp(sent)
  for token in doc:
    temp_tokens.append(token)
    if token.lemma_ not in uniq_tokens:
      uniq_tokens.append(token.lemma_)
  return temp_tokens, uniq_tokens

def count_cohesive_fx(sent, cohesive_list):
  #print(list_all_tokens)
  count_cohesive = 0
  #print(uni_cohesive_markers)
  for word in sent:
    if str(word).lower() in uni_cohesive_markers:
      count_cohesive += 1
  # print(count_cohesive)

  first_word = str(sent[0]).lower()
  for word in sent[1:]:
    bigram = []
    bigram.append(first_word)
    second_word = str(word).lower()
    bigram.append(second_word)
    #print(bigram)
    if bigram in bi_cohesive_markers:
      count_cohesive += 1
    first_word = second_word
  # print(count_cohesive)

  first_word = str(sent[0]).lower()
  second_word = str(sent[1]).lower()
  for word in sent[2:]:
    trigram = []
    trigram.append(first_word)
    trigram.append(second_word)
    third_word = str(word).lower()
    trigram.append(third_word)
    #print(trigram)
    if trigram in tri_cohesive_markers:
      count_cohesive += 1
    first_word = second_word
    second_word = third_word
  # print(count_cohesive)

  first_word = str(sent[0]).lower()
  second_word = str(sent[1]).lower()
  third_word = str(sent[2]).lower()
  for word in sent[3:]:
    quatgram = []
    quatgram.append(first_word)
    quatgram.append(second_word)
    quatgram.append(third_word)
    fourth_word = str(word).lower()
    quatgram.append(fourth_word)
    #print(quatgram)
    if quatgram == quat_cohesive_markers:
      count_cohesive += 1
    first_word = second_word
    second_word = third_word
    third_word = fourth_word
  # cfile.write(count_cohesive)
  # cfile.write('\n')
  cohesive_list.append(count_cohesive)
  return cohesive_list
  # print(count_cohesive)

  # print("COHESIVE MARKERS: " , count_cohesive)
  # cfile.close()

# count = 0
# clist = []

cfile = open('ttr.txt', 'w')


while True:
  line = reader.readline()
  if not line:
    break
  # line = line.translate(str.maketrans('', '', string.punctuation))
  line = re.sub(r"\s([?.!',](?:\s|$))", r'\1', line)
  sent_toks, uniq_toks = individual_sent_tokens(line)
  # clist = count_cohesive_fx(sent_toks, clist)
  ttr = len(uniq_toks) / len(sent_toks)
  cfile.write(str(ttr))
  cfile.write('\n')

cfile.close()
