
"""
*******************************************************************

Build the dependency graph features

*******************************************************************
"""

import argparse
import os

import torch
import torch.nn as nn
import torch_scatter
from transformers import BatchEncoding
from transformers import AutoTokenizer

from tqdm import tqdm
import pickle
import json

def get_all_dep_tags():
    tags_list=[]
    splits=['train','dev']
    for split in splits:
        samples=json.load(open('dep_parsed_{}.json'.format(split),'r'))
        for sample in tqdm(samples):
            for sent in sample['parsed_context']:
                tags_c=sent['rels']
                tags_list += tags_c
            for q in sample['qas']:
                for s in q['parsed_question']:
                    tags_q=s['rels']
                    tags_list += tags_q
    tags_list = list(set(tags_list))
    with open('./dep_rels_tag_list.txt','w') as f:
        for line in tags_list:
            f.write(line+'\n')
    return tags_list

def initialize_edge_dict():
    global deplabels
    edge_info = {}
    for each in deplabels:
        edge_info.setdefault(each,[])
    return edge_info

def load_parsed_results(data_split):
    results = json.load(open('dep_parsed_{}.json'.format(data_split),'r'))
    return results

#bert embeddings for the current sentence
#tokenized_text without special tokens
def add_word_nodes_and_create_subword_to_word_edges(tokenized_text,added_node_embeds=[],max_len=384,previous_token_num=0,previous_added_word_node_num=0):

    edge_index = []
    added_nodes = {}

    for i in range(len(tokenized_text.input_ids)):
        if BatchEncoding.word_to_tokens(tokenized_text,i) != None:
            start_token_pos = BatchEncoding.word_to_tokens(tokenized_text,i).start
            end_token_pos = BatchEncoding.word_to_tokens(tokenized_text,i).end

            if previous_token_num+end_token_pos<=max_len-1 and end_token_pos-start_token_pos>1:

                added_nodes[i] = len(added_nodes)+previous_added_word_node_num+max_len
                added_node_embeds += [(len(added_nodes)+previous_added_word_node_num+max_len-1,previous_token_num+start_token_pos+1,previous_token_num+end_token_pos+1)]
                for node_idx in range(start_token_pos,end_token_pos):
                    edge_index += [[node_idx+1+previous_token_num, len(added_nodes)+previous_added_word_node_num+max_len-1]]

    return added_nodes,added_node_embeds,edge_index

def get_dep_edges_for_sentence(sent_result,tokenized_text,added_nodes={},previous_token_num=0):
    words_num = max(BatchEncoding.words(tokenized_text))+1
    edges={}
    for i,rel in enumerate(sent_result['rels'][:words_num]):
        src=sent_result['arcs'][i]-1
        det=i
        if src==-1 or src>=words_num:
            continue

        src = added_nodes[src] if src in added_nodes else BatchEncoding.word_to_tokens(tokenized_text,src).start+previous_token_num+1
        det = added_nodes[det] if det in added_nodes else BatchEncoding.word_to_tokens(tokenized_text,det).start+previous_token_num+1

        edges.setdefault(rel,[])
        edges[rel] += [[src,det]]
    return edges

def update_edges(previous_edges,current_edges):
    for each in current_edges:
        previous_edges[each] += current_edges[each]
    return previous_edges

def update_nodes_by_max_len(original_result,tokenized_text,keep_token_num,max_len=384):
    updated_result={}
    keep_tokens = tokenized_text.input_ids[:keep_token_num+1]
    last_word_index = BatchEncoding.token_to_word(tokenized_text,keep_token_num)
    for each in original_result:
        updated_result[each]= original_result[each][:last_word_index]
    return updated_result

def pad_edges(edge_dict):
    for each in edge_dict:
        if edge_dict[each] == []:
            edge_dict[each] += [[0,0]]
    return edge_dict


def main(args):

    global tokenizer

    print('Load the dependency parsing results...')
    parsed_results = load_parsed_results(args.data_split)

    print('Begin to process...')

    print('Initialize or load the embedding for all all tags...')
    #initialize the embeddings of all tags
    #virtual nodes of same tag have the same initialization
    #tag_embeds_dict = initialize_tag_embeddings(args.emb_dim)

    max_len = args.max_len

    question_more_than_one_sent=[]

    for para_idx in tqdm(range(len(parsed_results))):
    #for para_idx in tqdm([1361]):

        for q_idx in range(len(parsed_results[para_idx]['qas'])):
            previous_token_num=0
            previous_added_word_node_num=0
            added_node_embeds=[]
            subword_to_word_edges=[]
            edge_info=initialize_edge_dict()

            parsed_q=parsed_results[para_idx]['qas'][q_idx]['parsed_question']

            tokenized_q=tokenizer(parsed_results[para_idx]['qas'][q_idx]['original_quesiton'],add_special_tokens=False)
            added_nodes,added_node_embeds,edge_index=add_word_nodes_and_create_subword_to_word_edges(tokenized_q,added_node_embeds,max_len,previous_token_num,previous_added_word_node_num)
            subword_to_word_edges += edge_index
            previous_added_word_node_num += len(added_nodes)

            for sent in parsed_q:
                dep_edges=get_dep_edges_for_sentence(sent,tokenized_q,added_nodes,previous_token_num)

                edge_info = update_edges(edge_info,dep_edges)
                previous_token_num += len(tokenized_q.input_ids)

            previous_token_num +=1
            for sent_idx in range(len(parsed_results[para_idx]['parsed_context'])):
                original_sent=parsed_results[para_idx]['original_context'][sent_idx]
                parsed_sent=parsed_results[para_idx]['parsed_context'][sent_idx]
                tokenized_sent=tokenizer(original_sent,add_special_tokens=False)
                added_nodes,added_node_embeds,edge_index=add_word_nodes_and_create_subword_to_word_edges(tokenized_sent,added_node_embeds,max_len,previous_token_num,previous_added_word_node_num)
                subword_to_word_edges += edge_index
                previous_added_word_node_num += len(added_nodes)
                if previous_token_num >=max_len-3:
                    break
                if previous_token_num+len(tokenized_sent.input_ids) > max_len-3:
                    keep_token_num = max_len-3-previous_token_num
                    parsed_sent=update_nodes_by_max_len(parsed_sent,tokenized_sent,keep_token_num,max_len)
                    dep_edges=get_dep_edges_for_sentence(parsed_sent,tokenized_sent,added_nodes,previous_token_num)
                    edge_info = update_edges(edge_info,dep_edges)
                    break
                dep_edges=get_dep_edges_for_sentence(parsed_sent,tokenized_sent,added_nodes,previous_token_num)
                edge_info = update_edges(edge_info,dep_edges)
                previous_token_num += len(tokenized_sent.input_ids)
            edge_info['subword2word'] = subword_to_word_edges
            if added_node_embeds != []:
                edge_info['added_node_embeds_mapping'] = added_node_embeds
            else:
                edge_info['added_node_embeds_mapping'] = [(0,0,0)]
            with open('{}/{}'.format(args.save_path,parsed_results[para_idx]['qas'][q_idx]['id']), "wb") as f:
                pickle.dump(pad_edges(edge_info), f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split','-d', default=None, required=True, type=str, help='dev or train')
    parser.add_argument('--emb_dim', default=768, type=int, help='dimension of node embedding')
    parser.add_argument('--max_len', default=384, type=int, help='max length of the input sequence')
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument('--save_path','-s', default="../data/squad_files/dependency_graphs", type=str, help='Path to save the generated graph data')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print('Initialize the pre-trained tokenizer of {}...'.format(args.model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    print('Initialize the basic edge dict by node type...')
    deplabels = [x.strip('\n') for x in open('./dep_rels_tag_list.txt','r').readlines()]
    deplabels.remove('root')
    main(args)
