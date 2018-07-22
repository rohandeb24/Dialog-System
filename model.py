# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.layers.core import Dense


def encoder(inputs, rnn_size, keep_prob, layers, vocab_size, encoder_embed_length, encoder_seq_length):
    '''Create an lstm cell with no. of nodes = rnn_size'''
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    '''apply dropout to each of the lstm cell'''
    lstm_cell_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob = keep_prob)
    '''Create Multiple Layers of RNN Cell of same rnn_size'''
    cell_layer = [lstm_cell_with_dropout]*layers
    encoder_cell = tf.contrib.rnn.MultiRNNCell(cell_layer)
    '''Create word embeddings'''
    encoder_embedings = tf.contrib.layers.embed_sequence(inputs, vocab_size, encoder_embed_length) #used to create embeding layer for the encoderr
    
    
    '''Creates a recurrent neural network specified by cell and returns A pair (outputs, state) where:
        outputs: The RNN output Tensor.
        state: The final state.
    '''
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell = encoder_cell, inputs =  encoder_embedings, sequence_length = encoder_seq_length, dtype = tf.float32)
    
    return encoder_outputs, encoder_states

def decoder_input(target, id_SOS,batch_size):
    
    return tf.concat([tf.fill([batch_size, 1] , id_SOS),tf.strided_slice(target,[0,0], [batch_size, -1], [1,1])],axis = 1)


def decoder_with_attention(rnn_size,encoder_outputs,encoder_seq_length, keep_probs):
    '''
    cell : basic lstm cell with dropout of size rnn_size
    
    attention_mechanism: Specifies BahdanauAttention mechanism with no of units (The depth of the query mechanism) = rnn_size
                         memory : the output of the encoder
                         memory_sequence_length : sequence lengths for the batch entriesin memory.  If provided, the memory tensor rows are masked with zeros
                                                 for values past the respective sequence lengths.
                                                 
    AttentionWrapper : Creates an attention wrapper on top of the RNN Cell with attention mechanism as 'Bahdanau'
                       
                       attention_layer_size: if attention_layer_size is not None, then it specifies the number of hidden units in a feedforward 
                                            layer within your decoder that is used to mix this context vector with the output of the decoder's 
                                            internal RNN cell to get the attention value. If attention_layer_size == None, it just uses the context 
                                            vector above as the attention value, and no mixing of the internal RNN cell's output is done.
    '''
    cell = tf.contrib.rnn.DropoutWrapper( tf.contrib.rnn.BasicLSTMCell(rnn_size),keep_probs)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units = rnn_size, memory = encoder_outputs,memory_sequence_length = encoder_seq_length)
    return tf.contrib.seq2seq.AttentionWrapper(cell = cell,attention_mechanism = attention_mechanism,attention_layer_size = rnn_size/2)



    
def decoder_embed(vocab_size,decoder_embed_size,decoder_inputs):
    
    embed_layer = tf.Variable(tf.random_uniform([vocab_size, decoder_embed_size]))
    embedings = tf.nn.embedding_lookup(embed_layer, decoder_inputs) 
    return embedings







def decoder_tr(decoder_inputs, enc_states, dec_cell, decoder_embed_size, vocab_size,
            dec_seq_len, max_seq_len, word_to_id, batch_size,
            embed_layer,embedings,output_layer):

	
    with tf.variable_scope('decoder'):
		
        train_helper = tf.contrib.seq2seq.TrainingHelper(embedings, 
                                                          dec_seq_len)
        

        train_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                                        train_helper, 
                                                        enc_states, 
                                                        output_layer)
        

        train_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, 
                                                                    impute_finished=True, 
                                                                    maximum_iterations=max_seq_len)
    return train_dec_outputs;
   
def decoder_infr(decoder_inputs, enc_states, dec_cell, decoder_embed_size, vocab_size,
            dec_seq_len, max_seq_len, word_to_id, batch_size,
            embed_layer,embedings,output_layer):
        
    with tf.variable_scope('decoder', reuse=True): #we use REUSE option in this scope because we want to get same params learned in the previouse 'decoder' scope

        starting_id_vec = tf.tile(tf.constant([word_to_id['<GO>']], dtype=tf.int32), [batch_size], name='starting_id_vec')
        

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embed_layer, 
                                                                    starting_id_vec, 
                                                                    word_to_id['<EOS>'])
        
        '''Takes the decoder layer, helper and the last encoding state and produces the output.
        The output layer is a fully connected layer that produces the probability vector'''
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper, 
                                                            enc_states, 
                                                            output_layer)
        
        '''
        returns the decoder outputs, final states and final sequence length
        impute_finished : If `True`, then states for batch
                            entries which are marked as finished get copied through and the
                            corresponding outputs get zeroed out.  
        '''
        inference_dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, 
                                                                       impute_finished=True, 
                                                                       maximum_iterations=max_seq_len)
        
    return inference_dec_output
    
    
    
    
    
    
    
    
def opt_loss(outputs, targets, dec_seq_len, max_seq_len, learning_rate, clip_rate):
    
    logits = tf.identity(outputs.rnn_output)
    
    mask_weigts = tf.sequence_mask(dec_seq_len, max_seq_len, dtype=tf.float32)
    
    with tf.variable_scope('opt_loss'):
        loss = tf.contrib.seq2seq.sequence_loss(logits, 
                                                targets, 
                                                mask_weigts)
        
        opt = tf.train.AdamOptimizer(learning_rate)

        gradients = tf.gradients(loss, tf.trainable_variables())
        clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_rate)
        traiend_opt = opt.apply_gradients(zip(clipped_grads, tf.trainable_variables()))
        
    return loss, traiend_opt
    
class ChatBott():

    '''
    1) rnn_size: output of a layer that is passes onto the next layer is a 'rnn_size' dimensional vector (
              h(t) = nonlinearity(U*x(t) + W*h(t-1))
              
              h(t)   ∈ R^m
              h(t-1) ∈ R^m
              x(t)   ∈ R^n
              U      ∈ R^mxn
              W      ∈ R^mxm
        then m = rnn_size
    
    2) learn_rate : learning rate of the optimization process.
    
    3) batch_size : size of each batch that is provided as input while training
    
    4) no_layers :  number of layers in the model
    
    5) vocabulary : a dictionary containing unique word to integer mapping
    
    6) inverse_vocab: a dictionary containing unique integer to word mapping
    
    7) encoder_embed_size : size of the encoder word embedding layer
    
    8) decoder_embed_size : size of the decoder word embedding layer
    
    9) clipping_rate - tolerance boundries for clipping gradients *************
    '''

	
   
    def __init__(self, rnn_size, learn_rate, batch_size, no_layers, vocabulary, encoder_embed_size, decoder_embed_size, clipping_rate):
        tf.reset_default_graph()
        
        
        self.batch_size = batch_size
       
       
        self.inputs = tf.placeholder(tf.int32,shape = [None,None],name = 'inputs')        
        self.targets = tf.placeholder(tf.int32,shape = [None,None],name = 'targets')
        self.keep_probs = tf.placeholder(tf.float32, name = 'dropout_rate') # or take it as an argument into the class
        
        self.encoder_sequence_length = tf.placeholder(tf.int32,(None,),name = 'encoder_sequence_length')
        self.decoder_sequence_length = tf.placeholder(tf.int32,(None,),name = 'decoder_sequence_length')
        max_sequence_length = tf.reduce_max(self.decoder_sequence_length,name = 'max_sequence_length')
        
        
        
  
        encoder_outputs, encoder_state = encoder(self.inputs, rnn_size, self.keep_probs, no_layers, len(vocabulary), encoder_embed_size, self.encoder_sequence_length)
        
        
        
        
        id_SOS = vocabulary['<GO>']
        dec_input = decoder_input(self.targets, id_SOS,batch_size)#decoder_input(self.targets, vocabulary['<SOS>'],self.batch_size)
        
        
        
        decoder = decoder_with_attention(rnn_size,encoder_outputs,self.encoder_sequence_length,  self.keep_probs)
        
        encoder_state_new = decoder.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=encoder_state[-1]) #The initial state of decoder
             
       
        
        embed_layer = tf.Variable(tf.random_uniform([len(vocabulary), decoder_embed_size]))
    	embedings = tf.nn.embedding_lookup(embed_layer, dec_input) 
    	output_layer = Dense(len(vocabulary), kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))
        
            
        train_outputs				 = decoder_tr(dec_input, 
                                                  encoder_state_new, 
                                                  decoder,
                                                  decoder_embed_size, 
                                                  len(vocabulary), 
                                                  self.decoder_sequence_length, 
                                                  max_sequence_length, 
                                                  vocabulary, 
                                                  batch_size,
                                                  embed_layer,embedings,output_layer)
        
        
        inference_output			 = decoder_infr(dec_input, 
                                                  encoder_state_new, 
                                                  decoder,
                                                  decoder_embed_size, 
                                                  len(vocabulary), 
                                                  self.decoder_sequence_length, 
                                                  max_sequence_length, 
                                                  vocabulary, 
                                                  batch_size,
                                                  embed_layer,embedings,output_layer)
        
        
        
        self.predictions  = tf.identity(inference_output.sample_id, name='preds')
        self.loss, self.opt = opt_loss(train_outputs, 
                                       self.targets, 
                                       self.decoder_sequence_length, 
                                       max_sequence_length,
                                       learn_rate, clipping_rate
                                       )
        
        
