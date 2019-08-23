# filter instances with event in (s, v, o) format
from nltk.parse import stanford
import sys
import os


def combine_text(graph, children_dict, root, exclude_children=[]):
    def combine_text_inner(children_dict, root, result, exclude_children):
        result.append(root)
        for i in children_dict[root]:
            if i not in exclude_children:
                combine_text_inner(children_dict, i, result, [])

    result = []
    combine_text_inner(children_dict, root, result, exclude_children)
    result = sorted(result)
    return ' '.join([graph.nodes[i]['word'] for i in result])


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = 'data/atomic/train_intent_react.txt'
        output_file= 'data/atomic/train_svo.txt'

    # os.environ['STANFORD_PARSER'] = '/usr/local/Cellar/stanford-parser/3.9.1/libexec/stanford-parser.jar'
    # os.environ['STANFORD_MODELS'] = '/usr/local/Cellar/stanford-parser/3.9.1/libexec/stanford-parser-3.9.1-models.jar'
    os.environ['STANFORD_PARSER'] = '/users4/kliao/data/stanford_nlp/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/users4/kliao/data/stanford_nlp/stanford-parser-3.9.1-models.jar'
    parser = stanford.StanfordDependencyParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

    lines = open(input_file, 'r').readlines()
    lines = [line.strip().split(' | ') for line in lines]
    output_file = open(output_file, 'w')

    for line in lines:
        event, intent, react = line

        verb = None
        subj = None
        obj = None

        graph = next(parser.raw_parse(event))
        children_dict = {}
        for i in graph.nodes:
            children_dict[i] = []
        for i in graph.nodes:
            node = graph.nodes[i]
            if node['head'] is not None:
                children_dict[node['head']].append(i)
        root = graph.nodes[children_dict[0][0]]
        if 'VB' in root['tag']:
            verb = children_dict[0][0]
            for i in children_dict[verb]:
                if graph.nodes[i]['rel'] == 'nsubj':
                    subj = i
                if graph.nodes[i]['rel'] == 'dobj':
                    obj = i
        # print(graph)
        # print(verb)
        # print(subj)
        # print(obj)
        # exit()

        if verb is not None and subj is not None and obj is not None:
            verb = combine_text(graph, children_dict, verb, [subj, obj])
            subj = combine_text(graph, children_dict, subj)
            obj = combine_text(graph, children_dict, obj)
            # print('subj:', subj)
            # print('verb:', verb)
            # print('obj: ', obj)
            # exit()
        else:
            continue

        output_file.write(' | '.join([subj, verb, obj, intent, react]) + '\n')
    output_file.close()
