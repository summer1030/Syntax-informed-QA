
"""
*******************************************************************

Do dependency parsing

Use the parser `Biaffine Parser' from SuPar.

More details about the parser can be found here,
https://github.com/yzhangcs/parser.

*******************************************************************
"""

import json
import argparse
from tqdm import tqdm
from supar import Parser


def get_all_samples(data_split):

    samples=json.load(open('../data/{}-v2.0.json'.format(data_split),'r'))

    all_paragraphs=[]
    for doc in tqdm(samples['data']):
        for para in doc['paragraphs']:
            tmp={}
            content = para['context']
            tmp['context'] = content
            tmp['qas'] = []
            for qa in para['qas']:
                foo={}
                foo['id'] = qa['id']
                foo['question'] = qa['question']
                foo['answers'] = []
                for ans in qa['answers']:
                    foo['answers'] += [ans['text']]
                tmp['qas'] += [foo]
            all_paragraphs += [tmp]

    with open('all_paragraphs_{}.json'.format(data_split),'w') as f:
        json.dump(all_paragraphs,f)

    return all_paragraphs

def get_all_ids(data_split):

    # para_ids = [int(x.strip('\n').split(' ')[0]) for x in open('./questions_more_than_one_sentences_{}.txt'.format(data_split),'r').readlines()]
    # qa_ids = [x.strip('\n').split(' ')[1] for x in open('./questions_more_than_one_sentences_{}.txt'.format(data_split),'r').readlines()]
    # return para_ids,qa_ids

    para_ids = [int(x.strip('\n').split(' ')[0]) for x in open('./have_problems_{}.txt'.format(data_split),'r').readlines()]
    return para_ids

def process_text(sent):
    return sent.strip(' ').strip('\n').replace('″','"').replace('…','...').replace('½','*').replace('\n',' ').replace('  ',' ').replace('´','\'').replace('ﬂ','f').replace('№','No')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', default='dev', type=str, help='dev or train')

    args = parser.parse_args()

    print('Initialize the model...')

    #initialize the dependency parsing model
    parser=Parser.load('biaffine-dep-en')

    print('Process the {} data'.format(args.data_split))
    # samples = get_all_samples(args.data_split)
    samples = json.load(open('./all_paragraphs_{}.json'.format(args.data_split),'r'))

    questions_more_than_one_sentences = []
    have_problems = []

    #Begin to parse data
    results=[]
    for i in tqdm(range(len(samples))):

        item = samples[i]
        tmp={}

        #content = item['context']
        content = [x.strip(' ')+'.' for x in item['context'].split('.') if x!=' ' and x!='']

        tmp['original_context'] = content
        tmp['parsed_context'] = []

        parsed_content = parser.predict(content, lang='en', prob=True, verbose=False)
        for sent in parsed_content:
            stored1 = {}
            stored1['arcs'] = sent.arcs
            stored1['rels'] = sent.rels
            tmp['parsed_context'] += [stored1]

        tmp['qas'] = []
        for q in item['qas']:
            stored2={}
            stored2['id']=q['id']
            stored2['original_quesiton']=q['question']
            stored2['parsed_question'] = []

            #q['question'] = q['question'].replace('″','"').replace('  ',' ')
            question = [x.strip(' ')+'.' for x in q['question'].split('.') if x!=' ' and x!='']
            parsed_question = parser.predict(question, lang='en', prob=True, verbose=False)

            for sent in parsed_question:
                stored3 = {}
                stored3['arcs'] = sent.arcs
                stored3['rels'] = sent.rels
                stored2['parsed_question'] += [stored3]

            tmp['qas'] += [stored2]

        results += [tmp]

    with open('dep_parsed_{}.json'.format(args.data_split),'w') as fout:
        json.dump(results,fout)


    # print('{} questions that contains more than one sentences.'.format(len(questions_more_than_one_sentences)))
    # #with open('questions_more_than_one_sentences_{}.txt'.format(args.data_split),'w') as f1:
    # #    f1.write('\n'.join('%s %s' % x for x in questions_more_than_one_sentences))

    # print('{} data have problems.'.format(len(have_problems)))
    # #with open('have_problems_{}.txt'.format(args.data_split),'w') as f2:
    # #    f2.write('\n'.join('%s' % x for x in have_problems))


if __name__ == '__main__':
    main()
