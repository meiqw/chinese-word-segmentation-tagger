
# coding: utf-8

# In[1]:


from itertools import chain
import nltk
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite

def sent2tag(sent):
    tagseq = list()
    for w in sent.split():
        tagseq.extend(word2tag(w))
    return tagseq

def word2tag(word):
    char_tags = list()
    if len(word) == 1:
        char_tags.append((word, 'S'))
    else:
        char_tags.append((word[0], 'B'))
        if len(word) > 2:
            for ch in word[1:-1]:
                char_tags.append((ch, 'M'))
        char_tags.append((word[-1], 'E'))
    return char_tags

def get_tagsequences(filepath, encoding='utf-8'):
    fh = open(filepath, 'r', encoding=encoding)
    return [sent2tag(sent) for sent in fh]

def word2features(sent, i):
    word = sent[i][0]
    features = [
        'bias',
        'word=' + word,
    ]
    prev1word = ''
    next1word = ''
    if i > 1:
        prev1word = sent[i-1][0]
        prev2word = sent[i-2][0]
        features.extend([
            'word-1=' + prev1word,
            'word-2=' + prev2word,
            'prevcurr=' + (prev1word+word),
            'prev2prev1='+(prev2word+prev1word),
            #'prevtags='+(sent[i-1][1]+sent[i-2][1])
        ])
    elif i > 0:
        prev1word = sent[i-1][0]
        features.extend([
            'word-1=' + prev1word,
            'BOS',
            'prevcurr=' + (prev1word+word),
            #not sure about this
            #'prevtags='+sent[i-1][1]
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-2:
        next1word = sent[i+1][0]
        next2word = sent[i+2][0]
        features.extend([
            'word+1=' + next1word,
            'word+2=' + next2word,
            'currnext=' + (word+next1word),
            'next1next2=' + (next1word+next2word)
        ])
    elif i < len(sent)-1:
        next1word = sent[i+1][0]
        features.extend([
            'word+1=' + next1word,
            'EOS',
            'currnext=' + (word+next1word)
        ])   
    else:
        features.append('EOS')
    
    features.append('prevnext='+(prev1word+next1word))
    #print(features)       
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2tags(sent):
    return [tag for word, tag in sent]

def sent2tokens(sent):
    return [word for word, tag in sent]

get_ipython().run_cell_magic('time', '', "train_sents = get_tagsequences('ctb2.0-train.seg')\nX_train = [sent2features(s) for s in train_sents]\ny_train = [sent2tags(s) for s in train_sents]\n\ntest_sents = get_tagsequences('ctb2.0-dev.seg')\nX_test = [sent2features(s) for s in test_sents]\ny_test = [sent2tags(s) for s in test_sents]")
get_ipython().run_cell_magic('time', '', 'trainer = pycrfsuite.Trainer(verbose=False)\n\nfor xseq, yseq in zip(X_train, y_train):\n    trainer.append(xseq, yseq)')
trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.params()
get_ipython().run_cell_magic('time', '', "trainer.train('cws.crfsuite')")
get_ipython().system('ls -lh ./cws.crfsuite')
trainer.logparser.last_iteration
print (len(trainer.logparser.iterations), trainer.logparser.iterations[-1])
tagger = pycrfsuite.Tagger()
tagger.open('cws.crfsuite')

example_sent = test_sents[0]
print(' '.join(sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("gold:  ", ' '.join(sent2tags(example_sent)))

def tag2seg(char_seq, tag_seq):
    """Given a raw sentence and a position tag sequence, generate the segmented word sequence"""
    assert len(char_seq) == len(tag_seq)
    segmented_sent = ''
    for i in range(len(char_seq)):
        if i != 0 and (tag_seq[i] == 'B' or tag_seq[i] == 'S'):
            segmented_sent = segmented_sent + " "
        segmented_sent = segmented_sent + char_seq[i]
    return segmented_sent

gold_seq = sent2tags(example_sent)
gold_segmentation = tag2seg(sent2tokens(example_sent), gold_seq)
predicted_seq = tagger.tag(sent2features(example_sent))
predicted_segmentation = tag2seg(sent2tokens(example_sent), predicted_seq)
print ("Gold segmentation: {}".format(gold_segmentation))
print ("Predicted segmentation: {}".format(predicted_segmentation))

def cws_classification_report(y_true, y_pred):
    """
    Classification report for a list of BMES-encoded sequences.
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag)
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

get_ipython().run_cell_magic('time', '', 'y_pred = [tagger.tag(xseq) for xseq in X_test]\n#print (y_pred[0])')
print(cws_classification_report(y_test, y_pred))

def count_per_sent(gold_seq, predicted_seq):
    correct = 0
    gold_start = 0
    predicted_start = 0
    gold_length = 0
    predicted_length = 0
    for i in range(len(gold_seq)):
        if gold_seq[i] == 'B':
            if gold_start == predicted_start:
                correct += 1
            gold_start = i
            gold_length += 1
            
        if predicted_seq[i] == 'B':
            predicted_start = i
            predicted_length += 1

        if gold_seq[i] ==  'S':
            if gold_start == predicted_start:
                correct += 1
            gold_start = i 
            gold_length += 1
                
        if predicted_seq[i] == 'S':
            predicted_start = i
            predicted_length += 1
            
    if gold_start == predicted_start:
        correct += 1
    gold_length += 1
    predicted_length += 1
    
    return correct,gold_length, predicted_length

def seg_eval(gold_labels, predicted_labels):
    correct_total = 0
    gold_total = 0
    predicted_total = 0
    #print(gold_labels)
    #print(predicted_labels)
    print(len(gold_labels), len(predicted_labels))
    for i in range(len(gold_labels)):
        gold_seq = gold_labels[i]
        predicted_seq = predicted_labels[i]
        #print(gold_seq)
        #print(predicted_seq)
        assert len(gold_seq) == len(predicted_seq)
        correct, gold_sent_length, predicted_sent_length = count_per_sent(gold_seq,predicted_seq)
        correct_total += correct
        gold_total += gold_sent_length
        predicted_total += predicted_sent_length
    precision = correct_total / predicted_total
    recall = correct_total / gold_total
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

prec, recall, f1 = seg_eval(y_test,y_pred)
print ("Precision: {}, recall: {}, f1-score: {}".format(prec, recall, f1))

from collections import Counter
info = tagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])

