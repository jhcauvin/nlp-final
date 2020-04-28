import spacy
#line 209 in data.py, prolly where we need to make the changes

nlp = spacy.load('en_core_web_sm')

passage = nlp("architecturally , the school has a catholic character . atop the main building 's gold dome is a golden statue of the virgin mary . immediately in front of the main building and facing it , is a copper statue of christ with arms upraised with the legend ' venite ad me omnes ' . next to the main building is the basilica of the sacred heart . immediately behind the basilica is the grotto , a marian place of prayer and reflection . it is a replica of the grotto at lourdes , france where the virgin mary reputedly appeared to saint bernadette soubirous in 1858 . at the end of the main drive ( and in a direct line that connects through 3 statues and the gold dome ) , is a simple , modern stone statue of mary .")

question = nlp("to whom did the virgin mary allegedly appear in 1858 in lourdes france ?")

candidates = []
eliminated = []
for sent in passage.sents:
    score = 0
    for token in sent:
        for word in question:
            if token.text == word.text and token.dep_ == word.dep_:
                score += 1
    print(score)
    if score > 3:
        candidates.append(sent)
    else:
        eliminated.append(sent)

print('num candidates', len(candidates))
print('num sentences in passage', len([x for x in passage.sents]))

print()

print('candidates', candidates)

print()

print('eliminated', eliminated)