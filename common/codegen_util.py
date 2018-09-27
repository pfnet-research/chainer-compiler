def format_code(lines, num_indents=0):
    formatted = []
    for i, line in enumerate(lines):
        num_indents -= line.count('}') * 4
        if line:
            ni = num_indents
            if line.endswith(':'):
                ni -= 4
            line = ' ' * ni + line
        formatted.append(line + '\n')
        if ('}' in line and '{' not in line and
            (i + 1 < len(lines) and '}' not in lines[i + 1])):
            formatted.append('\n')
        num_indents += line.count('{') * 4
    return formatted


def cond(conds, bodies):
    assert len(conds) + 1 == len(bodies)
    lines = []
    for i, c in enumerate(conds):
        line = ('if (%s) ' % c) + '{'
        if i:
            line = '} else ' + line
        lines.append(line)
        lines += bodies[i]
    lines.append('} else {')
    lines += bodies[len(conds)]
    return lines
