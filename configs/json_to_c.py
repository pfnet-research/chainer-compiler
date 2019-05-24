#!/usr/bin/python3

import json
import os
import re
import sys


def main():
    n = os.path.basename(sys.argv[1]).replace('.', '_')
    c = open(sys.argv[1]).read()
    # Strip comments.
    c = re.subn(r'//.*\n', '', c)[0]
    c = json.dumps(c)
    print('const char* %s = %s;' % (n, c))


main()
