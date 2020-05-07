import spacy
#line 209 in data.py, prolly where we need to make the changes

nlp = spacy.load('en_core_web_sm')

passage = nlp("architecturally , the school has a catholic character . atop the main building 's gold dome is a golden statue of the virgin mary . immediately in front of the main building and facing it , is a copper statue of christ with arms upraised with the legend ' venite ad me omnes ' . next to the main building is the basilica of the sacred heart . immediately behind the basilica is the grotto , a marian place of prayer and reflection . it is a replica of the grotto at lourdes , france where the virgin mary reputedly appeared to saint bernadette soubirous in 1858 . at the end of the main drive ( and in a direct line that connects through 3 statues and the gold dome ) , is a simple , modern stone statue of mary .")

question = nlp("to whom did the virgin mary allegedly appear in 1858 in lourdes france ?")
question_noun_subjs = [noun.text for noun in question if noun.dep_ == 'nsubj']
print(question_noun_subjs)
# print(test4)
sentences = [[sentence, 0] for sentence in passage.sents]
for sent in sentences:
    for word in sent[0]:
        if word.dep_ == 'nsubj' and word.text in question_noun_subjs:
            sent[1] += 1
for sentence in sentences:
    print(sentence[1])

# candidates = []
# eliminated = []
# for sent in passage.sents:
#     score = 0
#     for token in sent:
#         for word in question:
#             if token.text == word.text and token.dep_ == word.dep_:
#                 score += 1
#     print(score)
#     if score > 3:
#         candidates.append(sent)
#     else:
#         eliminated.append(sent)

# print('num candidates', len(candidates))
# print('num sentences in passage', len([x for x in passage.sents]))

# print()

# print('candidates', candidates)

# print()

# print('eliminated', eliminated)

# {"id": "", "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.", "qas": [{"answers": ["Saint Bernadette Soubirous"], "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?", "id": "5733be284776f41900661182", "qid": "38cc2597b6624bd8af1e8ba7f693096f", "question_tokens": [["To", 0], ["whom", 3], ["did", 8], ["the", 12], ["Virgin", 16], ["Mary", 23], ["allegedly", 28], ["appear", 38], ["in", 45], ["1858", 48], ["in", 53], ["Lourdes", 56], ["France", 64], ["?", 70]], "detected_answers": [{"text": "Saint Bernadette Soubirous", "char_spans": [[515, 540]], "token_spans": [[102, 104]]}]}, {"answers": ["a copper statue of Christ"], "question": "What is in front of the Notre Dame Main Building?", "id": "5733be284776f4190066117f", "qid": "b17a05e67fd14669860a380d66aed5fb", "question_tokens": [["What", 0], ["is", 5], ["in", 8], ["front", 11], ["of", 17], ["the", 20], ["Notre", 24], ["Dame", 30], ["Main", 35], ["Building", 40], ["?", 48]], "detected_answers": [{"text": "a copper statue of Christ", "char_spans": [[188, 212]], "token_spans": [[37, 41]]}]}, {"answers": ["the Main Building"], "question": "The Basilica of the Sacred heart at Notre Dame is beside to which structure?", "id": "5733be284776f41900661180", "qid": "80a511ed750842d08ecdfaaaa257d95f", "question_tokens": [["The", 0], ["Basilica", 4], ["of", 13], ["the", 16], ["Sacred", 20], ["heart", 27], ["at", 33], ["Notre", 36], ["Dame", 42], ["is", 47], ["beside", 50], ["to", 57], ["which", 60], ["structure", 66], ["?", 75]], "detected_answers": [{"text": "the Main Building", "char_spans": [[279, 295]], "token_spans": [[57, 59]]}]}, {"answers": ["a Marian place of prayer and reflection"], "question": "What is the Grotto at Notre Dame?", "id": "5733be284776f41900661181", "qid": "913477b8e7f84432a16e1594219815e5", "question_tokens": [["What", 0], ["is", 5], ["the", 8], ["Grotto", 12], ["at", 19], ["Notre", 22], ["Dame", 28], ["?", 32]], "detected_answers": [{"text": "a Marian place of prayer and reflection", "char_spans": [[381, 419]], "token_spans": [[76, 82]]}]}, {"answers": ["a golden statue of the Virgin Mary"], "question": "What sits on top of the Main Building at Notre Dame?", "id": "5733be284776f4190066117e", "qid": "1c969af40a3248eb87a6d8c9c7c8d4ad", "question_tokens": [["What", 0], ["sits", 5], ["on", 10], ["top", 13], ["of", 17], ["the", 20], ["Main", 24], ["Building", 29], ["at", 38], ["Notre", 41], ["Dame", 47], ["?", 51]], "detected_answers": [{"text": "a golden statue of the Virgin Mary", "char_spans": [[92, 125]], "token_spans": [[17, 23]]}]}], "context_tokens": [["Architecturally", 0], [",", 15], ["the", 17], ["school", 21], ["has", 28], ["a", 32], ["Catholic", 34], ["character", 43], [".", 52], ["Atop", 54], ["the", 59], ["Main", 63], ["Building", 68], ["'s", 76], ["gold", 79], ["dome", 84], ["is", 89], ["a", 92], ["golden", 94], ["statue", 101], ["of", 108], ["the", 111], ["Virgin", 115], ["Mary", 122], [".", 126], ["Immediately", 128], ["in", 140], ["front", 143], ["of", 149], ["the", 152], ["Main", 156], ["Building", 161], ["and", 170], ["facing", 174], ["it", 181], [",", 183], ["is", 185], ["a", 188], ["copper", 190], ["statue", 197], ["of", 204], ["Christ", 207], ["with", 214], ["arms", 219], ["upraised", 224], ["with", 233], ["the", 238], ["legend", 242], ["\"", 249], ["Venite", 250], ["Ad", 257], ["Me", 260], ["Omnes", 263], ["\"", 268], [".", 269], ["Next", 271], ["to", 276], ["the", 279], ["Main", 283], ["Building", 288], ["is", 297], ["the", 300], ["Basilica", 304], ["of", 313], ["the", 316], ["Sacred", 320], ["Heart", 327], [".", 332], ["Immediately", 334], ["behind", 346], ["the", 353], ["basilica", 357], ["is", 366], ["the", 369], ["Grotto", 373], [",", 379], ["a", 381], ["Marian", 383], ["place", 390], ["of", 396], ["prayer", 399], ["and", 406], ["reflection", 410], [".", 420], ["It", 422], ["is", 425], ["a", 428], ["replica", 430], ["of", 438], ["the", 441], ["grotto", 445], ["at", 452], ["Lourdes", 455], [",", 462], ["France", 464], ["where", 471], ["the", 477], ["Virgin", 481], ["Mary", 488], ["reputedly", 493], ["appeared", 503], ["to", 512], ["Saint", 515], ["Bernadette", 521], ["Soubirous", 532], ["in", 542], ["1858", 545], [".", 549], ["At", 551], ["the", 554], ["end", 558], ["of", 562], ["the", 565], ["main", 569], ["drive", 574], ["(", 580], ["and", 581], ["in", 585], ["a", 588], ["direct", 590], ["line", 597], ["that", 602], ["connects", 607], ["through", 616], ["3", 624], ["statues", 626], ["and", 634], ["the", 638], ["Gold", 642], ["Dome", 647], [")", 651], [",", 652], ["is", 654], ["a", 657], ["simple", 659], [",", 665], ["modern", 667], ["stone", 674], ["statue", 680], ["of", 687], ["Mary", 690], [".", 694]]}
