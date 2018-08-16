def format_code(lines):
    formatted = []
    num_indents = 0
    for line in lines:
        num_indents -= line.count('}') * 4
        if line:
            ni = num_indents
            if line.endswith(':'):
                ni -= 4
            line = ' ' * ni + line
        formatted.append(line + '\n')
        if '}' in line:
            formatted.append('\n')
        num_indents += line.count('{') * 4
    return formatted
