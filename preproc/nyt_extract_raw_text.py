import xml.dom.minidom
import re
import os
import sys
import glob


def extract_one_doc(filename):
    whitespace_pattern = re.compile(r'\s+')
    dom = xml.dom.minidom.parse(filename)
    nitf = dom.documentElement
    blocks = nitf.getElementsByTagName('block')
    text = []
    for block in blocks:
        if block.getAttribute('class') == 'full_text':
            ps = block.getElementsByTagName('p')
            for p in ps:
                line = p.childNodes[0].data
                line = whitespace_pattern.sub(' ', line)
                text.append(line)
    return text


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    month_dir = input_dir
    day_dirs = glob.glob(os.path.join(month_dir, '*'))
    day_dirs = sorted(day_dirs)
    for day_dir in day_dirs:
        if os.path.isdir(day_dir):
            dir_name = os.path.relpath(day_dir, input_dir)
            dir_name = os.path.join(output_dir, dir_name)
            os.makedirs(dir_name, exist_ok=True)
            docs = glob.glob(os.path.join(day_dir, '*'))
            docs = sorted(docs)
            for doc in docs:
                if os.path.isfile(doc):
                    text = extract_one_doc(doc)
                    output_file = open(os.path.join(dir_name, os.path.basename(doc).replace('.xml', '.txt')), 'w')
                    for line in text:
                        output_file.write(line + '\n')
                    output_file.close()
                    print(doc)
