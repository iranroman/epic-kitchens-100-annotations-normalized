import csv
import itertools
from itertools import compress
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re
import sys
from autocorrect import Speller
nltk.download('wordnet')  
nltk.download('omw-1.4')   

spell = Speller(lang='en')

# helper functions
def replace_right(source, target, replacement, replacements=None):
    return replacement.join(source.rsplit(target, replacements))

noun_classes_file = '/home/iran/train-CLIP-FT/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv'
verb_classes_file = '/home/iran/train-CLIP-FT/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
ann_file = '/home/iran/train-CLIP-FT/epic-kitchens-100-annotations/EPIC_100_validation.csv'

standardize_verb = True # NOTE: without remove_continuations==True not all verbs will be standardized
standardize_noun = True
remove_continuations = True
lemmatizer = WordNetLemmatizer()

raw_prepositions = ['-down','-in','-into','-onto','-with']
articles = ['a', 'an', 'the', 'all', 'some']

def get_class_dict(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        header = next(reader,None)
        return {v[1]:eval(v[2]) for v in reader}


noun_classes_dict = get_class_dict(noun_classes_file)
noun_classes_dict['surface'] = noun_classes_dict.pop('top')
noun_classes_dict['towel:paper'] = noun_classes_dict.pop('towel:kitchen')
noun_classes_dict['eggplant'] = noun_classes_dict.pop('aubergine')
noun_classes_dict['zucchini'] = noun_classes_dict.pop('courgette')
noun_classes_dict['the fire'] = noun_classes_dict.pop('hob')
noun_classes_dict['liquid:washing'].append('liquid:washing:op') # typo not addressed by noun classes
verb_classes_dict = get_class_dict(verb_classes_file)
verb_classes_dict['carry'].append('bring-into')
verb_classes_dict['drain'] = verb_classes_dict.pop('filter')
verb_classes_dict['put'] = [*verb_classes_dict['put'],*verb_classes_dict.pop('insert')]
                     
# load annotation file
with open(ann_file, newline='') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    header = next(reader,None)
    annotations = [l for l in reader]

first = int(sys.argv[1])

subs_dict = {}
# main loop
for i, l in enumerate(annotations[first:2608],start=first):
    print(i, l[2])
    print(l[8])
    l[8] = l[8].replace('.','')
    # remove articles
    words = l[8].split(' ')
    words = [word for word in words if word not in articles]
    # remove 'of'
    words_no_of = [word for i,word in enumerate(words[:-1]) if word!='of' and words[i+1]!='the']
    words_no_of.append(words[-1])
    l[8] = ' '.join(words_no_of)

    # remove types of oil
    l[8] = l[8].replace('olive oil', 'oil')
    l[8] = l[8].replace(' olive', '') if 'olive' in l[8] and 'oil' in l[8] else l[8]

    # help with plurals
    l[8] = l[8].replace('patties', 'pattys')

    # remove slow cooker bowl and pan
    l[8] = l[8].replace('slow cooker bowl', 'cooker bowl')
    l[8] = l[8].replace('slow cooker pan', 'cooker pan')

    # remove typos
    l[8] = l[8].replace(' washing op', '')
    l[8] = l[8].replace(' in to ', ' in ')
    l[8] = l[8].replace(' pate', ' plate')

    if l[8] not in subs_dict:
        raw_action = l[8]
        subs_dict[raw_action] = ''
        if remove_continuations:
            if 'continue' in l[8] or 'continuing' in l[8] or 'keep' in l[8]:
                gerund_verb = l[8].split()[1]
                infinitive_verb = lemmatizer.lemmatize(gerund_verb,'v')
                l[8] = l[8].replace('continue ','')
                l[8] = l[8].replace('continuing ','')
                l[8] = l[8].replace('keep ','')
                l[8] = l[8].replace(gerund_verb,infinitive_verb)
            elif 'still' in l[8]:
                gerund_verb = l[8].split()[1]
                infinitive_verb = lemmatizer.lemmatize(gerund_verb,'v')
                l[8] = l[8].replace('still ','',1)
                l[8] = l[8].replace(gerund_verb,infinitive_verb)
            elif l[8][:3]=='and' or l[8][:3]=='now':
                l[8] = l[8][4:]
        if standardize_noun:
            raw_nouns = eval(l[13])
            standard_nouns = [list(compress(list(noun_classes_dict.keys()),[raw_noun in nouns for nouns in noun_classes_dict.values()]))[0] for raw_noun in raw_nouns]
            assert [standard_noun in noun_classes_dict.keys() for standard_noun in standard_nouns]
            for raw_noun, standard_noun in zip(raw_nouns, standard_nouns):
                if ':' in raw_noun:
                    noun_words = raw_noun.split(':')
                    for w in noun_words[1:]:
                        if '{}s'.format(w) in l[8]:
                            l[8] = l[8].replace(' {}s'.format(w),'',1) # replace only one occurrence
                        else:
                            l[8] = l[8].replace(' {}'.format(w),'',1) # replace only one occurrence
                    raw_noun = noun_words[0]
                if raw_noun not in l[8]:
                    l[8] = '{} {}'.format(l[8],raw_noun)
                standard_noun = ' '.join(reversed(standard_noun.split(':'))) if ':' in standard_noun else standard_noun
                l[8] = l[8].replace(raw_noun, standard_noun, 1)
        if standardize_verb:
            raw_verb = l[9]
            first_word = l[8].split()[0]
            if 'ing' in first_word: # remove gerunds
                infinitive_verb = lemmatizer.lemmatize(first_word,'v')
                l[8] = l[8].replace(first_word, infinitive_verb,1)
            standard_verb = list(compress(list(verb_classes_dict.keys()),[raw_verb in verbs for verbs in verb_classes_dict.values()]))[0]
            assert standard_verb in verb_classes_dict.keys()
            if any([raw_prep in raw_verb for raw_prep in raw_prepositions]):
                verb, prep = raw_verb.split('-') 
                l[8] = l[8].replace(' {}'.format(prep),'')
                l[8] = l[8].replace(verb,raw_verb,1)
                l[8] = l[8].replace(raw_verb, standard_verb,1)
                if prep=='down':
                    words = l[8].split(' ')
                    standard_verb_idx = words.index(standard_verb)
                    words.insert(standard_verb_idx+1,prep)
                    l[8] = ' '.join(words)
                else:
                    standard_noun = ' '.join(list(reversed(standard_nouns[-1].split(':')))) if ':' in standard_nouns[-1] else standard_nouns[-1]
                    l[8] = replace_right(l[8],standard_noun, '{} {}'.format(prep, standard_noun), 1)
                #words.insert(standard_verlen(object_noun)+1,prep)
                #l[8] = ' '.join(words)
                #l[8] = l[8].replace(verb_object,' '.join([verb_object, prep]))
            else:
                l[8] = l[8].replace(raw_verb, standard_verb,1)
            
            # remove any duplicated spaces as last step
            l[8] = re.sub(' +', ' ', l[8])

        # remove dashes
        l[8] = l[8].replace('-',' ')
        # remove duplicate consecutive words
        l[8] = ' '.join(k for k, _ in itertools.groupby(l[8].split()))
        subs_dict[raw_action]=l[8]
        print(subs_dict[raw_action])
        if subs_dict[raw_action] == 'blahblah add water into bowl':
            print(raw_action)
            input()
        try:
            if subs_dict[raw_action] == raw_action:
                continue
            print('\n')
            input()
        except KeyboardInterrupt:
            actions = list(set(list(subs_dict.values())))
            actions.sort()
            for a in actions:
                print(''.join([w+'\t\t' for w in a.split()]))
            print(len(set(list(subs_dict.keys()))), len(actions))
            input()
    else:
        l[8] = subs_dict[l[8]]
        print(l[8]+'\n')
actions = list(set(list(subs_dict.values())))
actions.sort()
for a in actions:
    print(''.join([w+'\t\t' for w in a.split()]))
print(len(set(list(subs_dict.keys()))), len(actions))
input()
annotations.insert(0,header)
with open("EPIC_100_validation_normalized.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerows(annotations)
actions = list(set(list(subs_dict.values())))
actions.sort()
actions_counted = [(list(subs_dict.values()).count(a), a) for a in actions]
actions_counted.sort()
for a in actions_counted:
    #if a[0]==1:
    #    print(a[0],list(subs_dict.keys())[list(subs_dict.values()).index(a[1])])
    #else:
    print(a[0],[list(subs_dict.keys())[i] for i, v in enumerate(list(subs_dict.values())) if v == a[1]], a[1])
    #input()
    #print(list(subs_dict.values()).count(a),''.join([w+'\t\t' for w in a.split()]))
print(len(set(list(subs_dict.keys()))), len(actions))
