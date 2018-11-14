#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:11:20 2018

@author: Karthick Aravindan
"""
#Import the libraries

import numpy as np
import tensorflow as tf
import re
import time

# Import dataset
lines = open('movie_lines.txt', encoding ='utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt',encoding ='utf-8',errors = 'ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))

questions =[]
answers =[]
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])  
        
# Cleaning text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's' ","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll","will",text)
    text = re.sub(r"\'ve","have",text)
    text = re.sub(r"\'re","are",text)
    text = re.sub(r"\'d","would",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","can not",text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text) 
    #print(text)
    return text


clean_question = []
for question in questions:
    clean_question.append(clean_text(question))
    

clean_answer = []
for answer in answers:
    clean_answer.append(clean_text(answer))
        
        
# Creating a dict that maps each words to its number of occurance
word2count = {}
for question in clean_question:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1


for answer in clean_answer:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1

                

# Mapping questions and answer to unqiue integer 

threshold =20
questionwords2int ={}
word_number =0
for word,count in word2count.items():
    if count >= threshold:
        questionwords2int[word]= word_number
        word_number +=1

answerwords2int ={}
word_number =0
for word,count in word2count.items():
    if count >= threshold:
        answerwords2int[word]= word_number
        word_number +=1


# Adding the last tokens to these two dict
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionwords2int[token] =len(questionwords2int)+1
for token in tokens:
    answerwords2int[token] =len(answerwords2int)+1
    
# Inverse dictinoary of answersword2int 
answerint2words = {w_i:w for w,w_i in answerwords2int.items()}

# Adding end of string to every answer
for i in range(len(clean_answer)):
    clean_answer[i] += ' <EOS>'

# Translating all the question and answer to integer 

question_into_int = []
for question in clean_question:
    ints=[]
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    question_into_int.append(ints)
    

answer_into_int = []
for answer in clean_answer:
    ints=[]
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answer_into_int.append(ints)

# Sorting question and answer by length of question

sorted_clean_question =[]
sorted_clean_answer =[]
for length in range(1,25+1):
    for i in enumerate(question_into_int):
        if len(i[1]) == length:
            sorted_clean_question.append(question_into_int[i[0]])
            sorted_clean_answer.append(answer_into_int[i[0]])



# Sequence to sequence model
            

# Creating placeholder function
def model_input():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'input')
    targets = tf.placeholder(tf.int32, [None,None], name = 'target')
    lr = tf.placeholder(tf.float32,  name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    return inputs,targets,lr,keep_prob
    
# Preprocessing the target
def preprocess_targets(targets,word2int,batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    
    return preprocess_targets               

# creating the Encoder RNN Layer

def encoder_rnn(rnn_inputs,rnn_size,rnn_layers,keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob= keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    encoder_output,encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                      cell_bw = ecoder_cell,
                                                      sequence_length = sequence_length,
                                                      inputs =rnn_inputs,
                                                      dtype = tf.float32)
    return encoder_state

# Decoding RNN layer training set

def decode_training_set(encoder_state,decoder_cell,
                        decoder_embedded_input,sequence_length,decoding_scope,
                        output_function,keep_prob,batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construction_function= tf.contrib.seq2seq.prepare_attention(attention_states,attention_option ='bahdanau',num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_fnction,
                                                                              name ="attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope=decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the test/validation set


def decode_test_set(encoder_state,decoder_cell,
                        decoder_embeddings_matrix,sequence_length,
                        sos_id,eos_id,maximum_length,num_words,
                        decoding_scope,
                        output_function,keep_prob,batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construction_function= tf.contrib.seq2seq.prepare_attention(attention_states,attention_option ='bahdanau',num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_fnction,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,        
                                                                              name ="attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope=decoding_scope)
    return test_predictions













































      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        