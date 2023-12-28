import spacy
import re
# import nltk
# from nltk import bigrams
# from nltk import ngrams

nlp = spacy.load("en_core_web_sm")

dist_dict = {}
list_all_tokens = []
list_unique_tokens = []
list_pos_tags = []

uni_cohesive_markers = ["because", "besides", "but", "consequently", "despite", "except", "further", "furthermore", "hence", "however", "instead", "maybe", "moreover", "nevertheless", "otherwise",  "since", "so", "therefore",  "though", "thus",  "yet", "concerning"]
bi_cohesive_markers = [["as","for"], ["as","to"], ["even", "if"], ["even","though"], ["in", "addition"], ["in","conclusion"], ["in","spite"], ["referring","to"], ["the", "former"], ["the", "latter"], ["this", "implies"]]
tri_cohesive_markers = [["in","other","words"], ["is","to","say"], ["on","account","of"], ["on","the","contrary"], ["with","reference","to"], ["with","regard","to"]]
quat_cohesive_markers = ["on","the","other","hand"]

# Universal Dependency Average Summed Distance

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

reader = open('para_t5.txt', 'r')

def collect_all_unique_tokens(sent):
  doc = nlp(sent)
  for token in doc:
    list_all_tokens.append(token)
    list_pos_tags.append(token.pos_)
    if token.lemma_ not in list_unique_tokens:
      list_unique_tokens.append(token.lemma_)

# count = 0
while True:
  line = reader.readline()
  if not line:
    break
  #line = line.translate(str.maketrans('', '', string.punctuation))
  line = re.sub(r"\s([?.!',](?:\s|$))", r'\1', line)
  collect_all_unique_tokens(line)
  #print(line)
  try:
    summed_dist(line)
  except:
    print("An exception occurred")
  # count += 1
  # if count%5000==0:
  #   print("Now at line: ", str(count))
  # file.write(str(dist))
  # file.write('\n')

# Cohesive Markers

#print(list_all_tokens)
count_cohesive = 0
#print(uni_cohesive_markers)
for word in list_all_tokens:
  if str(word).lower() in uni_cohesive_markers:
    count_cohesive += 1
print(count_cohesive)

first_word = str(list_all_tokens[0]).lower()
for word in list_all_tokens[1:]:
  bigram = []
  bigram.append(first_word)
  second_word = str(word).lower()
  bigram.append(second_word)
  #print(bigram)
  if bigram in bi_cohesive_markers:
    count_cohesive += 1
  first_word = second_word
print(count_cohesive)

first_word = str(list_all_tokens[0]).lower()
second_word = str(list_all_tokens[1]).lower()
for word in list_all_tokens[2:]:
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
print(count_cohesive)

first_word = str(list_all_tokens[0]).lower()
second_word = str(list_all_tokens[1]).lower()
third_word = str(list_all_tokens[2]).lower()
for word in list_all_tokens[3:]:
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
print(count_cohesive)

print("COHESIVE MARKERS: " , count_cohesive)

# Lexical Richness (TTR)

ttr = len(list_unique_tokens) / len(list_all_tokens) # lexical richness / type-to-token ratio (TTR): dividing the number of unique (lemmatized) tokens by the total number of tokens
print(ttr)

# count_propn = list_pos_tags.count('PROPN')
# print(count_propn)
count_coord = list_pos_tags.count('CCONJ') + list_pos_tags.count('SCONJ')
print(count_coord)

pos_counts = {}

for i in list_pos_tags:
  pos_counts[i] = pos_counts.get(i, 0) + 1

pos_sort = sorted(pos_counts.items(), key=lambda x: x[0])

with open('pos_counts.txt', 'w') as pf:
  for pos,num in pos_sort:
    out = str(pos) + "," + str(num)
    pf.write(out)
    pf.write("\n")

pf.close()

def average(lst):
  return sum(lst) / len(lst)

res = sorted(dist_dict.items(), key=lambda x: x[0])

print(dist_dict)
print(res)

ud_avg_L = [average(x[1]) for x in res]
#print(ud_avg_L)
sent_length_L = [x[0] for x in res]
#print(sent_length_L)

with open('sent_length.txt', 'w') as fp:
    for item in sent_length_L:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

fp.close()

with open('ud_avg.txt', 'w') as ud:
    for item in ud_avg_L:
        # write each item on a new line
        ud.write("%s\n" % item)
    print('Done')

ud.close()

reader.close()
