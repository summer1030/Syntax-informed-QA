
"""
*******************************************************************

Do consistuency parsing

Use the parser `Berkeley Neural Parser' from spaCy.

More details about the parser can be found here,
https://spacy.io/universe/project/self-attentive-parser,
https://github.com/nikitakit/self-attentive-parser.

*******************************************************************
"""

import json
import argparse
from tqdm import tqdm
import benepar, spacy

def get_all_ids(data_split):

    # doc_ids = [int(x.strip('\n').split(' ')[0]) for x in open('./to-parse-{}.txt'.format(data_split),'r').readlines()]
    # qa_ids = [x.strip('\n').split(' ')[1] for x in open('./to-parse-{}.txt'.format(data_split),'r').readlines()]

    to_do_list = [x.strip('\n') for x in open('./to-parse-{}.txt'.format(data_split),'r').readlines()]
    return to_do_list

    # para_ids = [int(x.strip('\n').split(' ')[0]) for x in open('./to-parse-{}.txt'.format(data_split),'r').readlines()]
    # return para_ids

def process_text(sent):
    return sent.strip(' ').strip('\n').replace('″','"').replace('…','...').replace('½','*').replace('\n',' ').replace('  ',' ').replace('´','\'').replace('ﬂ','f').replace('№','No')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split','-d', default='dev', type=str, help='dev or train')

    args = parser.parse_args()

    print('Initialize the model...')
    spacy.prefer_gpu()

    #initialize the consistency parsing model
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})


    print('Process the {} data'.format(args.data_split))
    samples = json.load(open('../data/{}-v2.0.json'.format(args.data_split),'r'))['data']

    to_do_list = get_all_ids(args.data_split)
    #para_ids = get_all_ids(args.data_split)

    questions_more_than_one_sentences = []
    have_problems = []

    #Begin to parse data
    results={}
    for each in tqdm(to_do_list):

        doc_id, para_id, qa_id, qid = int(each.split(' ')[0]),int(each.split(' ')[1]),int(each.split(' ')[2]),each.split(' ')[3]

        doc = samples[doc_id]
        para = doc['paragraphs'][para_id]
        qa = para['qas'][qa_id]


        item = para
        tmp={}

        #content = item['context']
        content = [x.strip(' ')+'.' for x in para['context'].split('.') if x!=' ' and x!='']

        tmp['original_context'] = content
        tmp['parsed_context'] = []

        try:
            doc = nlp(content)
            for sent in list(doc.sents):
                tmp['parsed_context'] += [sent._.parse_string]
        except:

            for sent in content:
                try:
                    tmp['parsed_context'] += [list(nlp(sent).sents)[0]._.parse_string]
                except:
                    try:
                        tmp['parsed_context'] += [list(nlp(process_text(sent)).sents)[0]._.parse_string]
                    except:
                        tmp['parsed_context'] += [list(nlp(process_text(sent[:-1])).sents)[0]._.parse_string]


        tmp['original_quesiton']=qa['question']
        tmp['parsed_question'] = []

        processed_q = qa['question'].replace('″','"').replace('  ',' ')

        if len(list(nlp(processed_q.strip(' ')).sents)) > 1:

            for id_ in range(len(list(nlp(processed_q.strip(' ')).sents))):
                tmp['parsed_question'] += [list(nlp(processed_q.strip(' ')).sents)[id_]._.parse_string]
        else:
            tmp['parsed_question'] += [list(nlp(processed_q.strip(' ')).sents)[0]._.parse_string]


        results[qid] = tmp


    with open('con_parsed_{}.json'.format(args.data_split),'w') as fout:
        json.dump(results,fout)


    #print('{} questions that contains more than one sentences.'.format(len(questions_more_than_one_sentences)))
    #with open('questions_more_than_one_sentences_{}.txt'.format(args.data_split),'w') as f1:
    #    f1.write('\n'.join('%s %s' % x for x in questions_more_than_one_sentences))

    #print('{} data have problems.'.format(len(have_problems)))
    #with open('have_problems_{}.txt'.format(args.data_split),'w') as f2:
    #    f2.write('\n'.join('%s' % x for x in have_problems))


if __name__ == '__main__':
    main()
