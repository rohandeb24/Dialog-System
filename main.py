import tensorflow as tf
import numpy as np
import hyperparams
from model import ChatBott
import Data_preprocessing
from tqdm import tqdm
from sklearn.metrics import accuracy_score

model = ChatBott(hyperparams.RNN_SIZE,
		             hyperparams.LEARNING_RATE,
		             hyperparams.BATCH_SIZE, 
		             hyperparams.NUM_LAYERS,
		             Data_preprocessing.vocabulary,
		             hyperparams.ENCODER_EMBED_SIZE,
		             hyperparams.DECODER_EMBED_SIZE,
		             hyperparams.CLIP_RATE)


def accuracy(target, logits):
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(target,[(0,0),(0,max_seq - target.shape[1])],'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(logits,[(0,0),(0,max_seq - logits.shape[1])],'constant')
    target = np.ndarray.flatten(target)
    logits = np.ndarray.flatten(logits)
    return accuracy_score(target, logits)


saver = tf.train.Saver(max_to_keep=10)
def new_session():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    return session


def restore_session():
    sess = tf.Session();
    s = tf.train.import_meta_graph(str(tf.train.latest_checkpoint('./checkpoint'))[2:]+'.meta')
    s.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
    return sess		             
		             		             

def train(session):
    for i in range(hyperparams.EPOCHS):
        epoch_accuracy = []
        epoch_loss = []
        for b in range(len(hyperparams.BUCKETS)):
            
            bucket = Data_preprocessing.get_bucket(b)
            bucket_accuracy = []
            bucket_loss = []
            
            questions = []
            answers = []
            
            for k in range(len(bucket)):
                questions.append(np.array(bucket[k][0]))
                answers.append(np.array(bucket[k][1]))
                
            for j in tqdm(range(len(bucket) //  hyperparams.BATCH_SIZE)):
                            
    
                
                X_batch = questions[i*hyperparams.BATCH_SIZE:(i+1)*hyperparams.BATCH_SIZE]
                Y_batch = answers[i*hyperparams.BATCH_SIZE:(i+1)*hyperparams.BATCH_SIZE]
                
                feed_dict = {model.inputs:X_batch, 
 		                 model.targets:Y_batch, 
 		                 model.keep_probs:hyperparams.KEEP_PROBS, 
 		                 model.decoder_sequence_length:[len(Y_batch[0])]*hyperparams.BATCH_SIZE,
 		                 model.encoder_sequence_length:[len(X_batch[0])]*hyperparams.BATCH_SIZE}
                
                cost, _, preds = session.run([model.loss, model.opt, model.predictions], feed_dict=feed_dict)
                
                epoch_accuracy.append(accuracy(np.array(Y_batch), np.array(preds)))
                bucket_accuracy.append(accuracy(np.array(Y_batch), np.array(preds)))
                
                bucket_loss.append(cost)
                epoch_loss.append(cost)
                
            
            
        print("EPOCH: {}/{}".format(i+1, hyperparams.EPOCHS), " | Epoch loss: {}".format(np.mean(epoch_loss)), " | Epoch accuracy: {}".format(np.mean(epoch_accuracy)))
        
        saver.save(session, "checkpoint/chatbot_{}.ckpt".format(i))


def predict(sen,sess):
   
    sen = Data_preprocessing.clean(sen)
    sen_vec = Data_preprocessing.convert_sen_to_vector(sen) 
    X_batch = Data_preprocessing.create_batch(sen_vec,hyperparams.BATCH_SIZE)
    feed_dict = {model.inputs:X_batch, 
  	                     model.keep_probs:1, 
  	                     model.decoder_sequence_length:[20]*hyperparams.BATCH_SIZE,
  	                     model.encoder_sequence_length:[len(X_batch[0])]*hyperparams.BATCH_SIZE}
    
    output_batch = sess.run([model.predictions], feed_dict=feed_dict)
    output = output_batch[0]
    
    return output
