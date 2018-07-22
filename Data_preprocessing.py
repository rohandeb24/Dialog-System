import re
import pickle
import hyperparams
import numpy as np
THRESHOLD = 20
SEQLEN = 25

'''Opens the file and returns
   a list where each element 
   is a line from the file'''
def readline(filename):
    return open(filename,'r').read().split("\n")[:-1]



'''takes in input a list containing lines 
    from movie_lines.txt and returns a
    dictionary with key = sentence_id ([0])
    and value = sentence ([4])
    eg  {"L1045": "They do not!"} ''' 
def line_to_id(line):
    sentence = {}
    for x in line:
        _line = x.split(" +++$+++ ")
        #print _line
        sentence[_line[0]] = _line[4]
    return sentence

'''takes in input a list containing lines 
    from movie_conversations.txt and 
    returns a list of conversations([3])
    eg ['L194', 'L195', 'L196', 'L197']'''
def list_of_conversations(line):
    conversations = []
    for _line in line:
        conversations.append(_line.split(" +++$+++ ")[3][1:-1])
    return conversations

'''takes in input a list containg various conversations,
    each being a series of sentences and returns two lists 
    'question' and 'answer' containing ids of sentences'''
def question_answer(list_of_conversations):
    question = []
    answer = []
    for _line in list_of_conversations:
        _cnvr = _line.split(", ") #_cnvr[0]: ["'L194'", "'L195'", "'L196'", "'L197'"]
        for i in range(len(_cnvr)-1):
            
            question.append(_cnvr[i][1:-1]) # 'L194' 'L195' 'L196' '''
            answer.append(_cnvr[i+1][1:-1]) # 'L195' 'L196' 'L197' '''
    return question, answer


def clean(text):
    text = text.lower()
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    #text = re.sub(r"\.", " .", text)
    text = re.sub(r"\?", " ?", text)
    text = re.sub(r"!", " !", text)
    text = re.sub(r"/", " /", text)
    text = re.sub(r",", "", text)
    text = re.sub(r'"', ' "', text)
    text = re.sub(r"-", " -", text)
    text = re.sub(r'  ', ' ', text)
	

    text = re.sub(r"[.-<>{}+=|'()\:@]", "", text)
    return text
def clean_sentence(sentence):
    lines = sentence.items()
    
    for idx,text in lines:
        #add more
        text = clean(text)
        
    	sentence[idx] = text
    return sentence


def create_vocab(sentence):
    word_count = {}
    lines = sentence.items()
    for idx,text in lines: 
        words = text.split(" ")
        for word in words:
            if word_count.has_key(word):
                word_count[word] += 1
            else:
                word_count[word] = 1
    vocabulary = {}
    i = 0
    word_items = word_count.items()
    for word, count in word_items:
        if word_count[word] >= THRESHOLD:
            vocabulary[word] = i
            i +=1
    vocabulary['<EOS>'] = i
    vocabulary['<GO>'] = i+1
    vocabulary['<PAD>'] = i+2
    vocabulary['<UNK>'] = i+3
    
    inverse_vocab ={}
    for key, val in vocabulary.items():
       inverse_vocab[val] = key
    
    return vocabulary, inverse_vocab


def convert_sen_to_vector(sen):
    sen = clean(sen)
    words = sen.split(" ")
    vector = []
    for word in words:
        if word == '' or word == ' ':
            continue
        elif vocabulary.has_key(word):
            vector.append(np.int32(vocabulary[word]))
        else:
            vector.append(np.int32(vocabulary['<OUT>']))
    return vector

def sentence_vector(question, answer):
    question_vector = []
    answer_vector = []
    
    for i in range(len(question)):
        question_words = sentence[question[i]].split(" ")
        vector = []
        for word in question_words:
            if word == '' or word == ' ':
                continue
            elif vocabulary.has_key(word):
                vector.append(np.int32(vocabulary[word]))
            else:
                vector.append(np.int32(vocabulary['<UNK>']))
        if vector == []:
            continue
        question_vector.append(vector)

        answer_words = sentence[answer[i]].split(" ")
        vector = []
        for word in answer_words:
            if word == '' or word == ' ':
                continue
            elif vocabulary.has_key(word):
                vector.append(np.int32(vocabulary[word]))
            else:
                vector.append(np.int32(vocabulary['<UNK>']))
        
        vector.append(vocabulary['<EOS>'])
        answer_vector.append(vector)
    return question_vector, answer_vector



                


movie_lines = readline("movie_lines.txt")
movie_conversations = readline("movie_conversations.txt")


sentence = line_to_id(movie_lines) #dictionary of (id,sentence)
list_of_conversations = list_of_conversations(movie_conversations)

question, answer = question_answer(list_of_conversations)
sentence = clean_sentence(sentence)

vocabulary, inverse_vocab = create_vocab(sentence)

question_vector, answer_vector = sentence_vector(question,answer)


def pad_questions(question, pad, length):
	return [pad] * (length - len(question)) + question


def pad_answers(answer, pad, length):
	return answer + [pad] * (length - len(answer))
	
	
def create_bucket(buckets_list):
	already_seen_data = []
	buckets = []
	for bucket in buckets_list:
		bucketed_data = []
		encoder_len = bucket[0]
		decoder_len = bucket[1]
		for i in range(len(question_vector)):
			if len(question_vector[i]) <= encoder_len and len(answer_vector[i]) <= decoder_len:
				if i not in already_seen_data:
					padded_ques = pad_questions(question_vector[i], vocabulary['<PAD>'], encoder_len)
					padded_ans  = pad_answers  (answer_vector[i], vocabulary['<PAD>'], decoder_len)
					
					bucketed_data.append((padded_ques, padded_ans))
					already_seen_data.append(i)

		buckets.append(bucketed_data)

	return buckets

buckets = create_bucket(hyperparams.BUCKETS)

def get_bucket(i):
	return buckets[i]

   
def create_batch(x,batch_size):
        x = np.array(x)
        y = np.array(([0]*len(x),)*(batch_size-1))
	return np.vstack((x,y))
    
    
