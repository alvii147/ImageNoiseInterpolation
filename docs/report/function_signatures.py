from pathlib import Path

PREFIX_COLOR = 'blue'
NAME_COLOR = 'orange'
PARAM_COLOR = 'darkblue'
PARAM_VALUE_COLOR = 'red'
SIGNATURE_FONT_SIZE = '15pt'
MAIN_FONT_FAMILY = 'monospace'
MAIN_FONT_WEIGHT = 'bold'

def addSpanElement(element, style_props):
    style_str = ''
    for prop in style_props:
        style_str += f'{prop[0]}: {prop[1]}; '
    style_str = style_str.strip()

    span_open = f'<span style="{style_str}">'
    span_close = '</span>'

    return span_open + element + span_close

def funcToHTML(signature, prefix):
    fn = signature.replace('def', '').strip()
    fn = fn.rstrip(':')
    fn = fn.rstrip(')')
    name, params = [i.strip() for i in fn.split('(')]
    params = [i.strip() for i in params.split(',')]
    params = [i.split('=') for i in params]

    html_prefix = prefix + '.'
    html_prefix = addSpanElement(html_prefix, [('color', PREFIX_COLOR)])

    html_name = addSpanElement(name, [('color', NAME_COLOR)])

    html_params = []
    for param in params:
        html_param = addSpanElement(param[0], [('color', PARAM_COLOR)])
        if len(param) > 1:
            html_param += '=' + addSpanElement(
                param[1],
                [('color', PARAM_VALUE_COLOR)]
            )
        html_params.append(html_param)

    html_params = ', '.join(html_params)

    html = addSpanElement(
        html_prefix + addSpanElement(
            html_name + '(' + html_params + ')',
            [('font-size', SIGNATURE_FONT_SIZE)]
        ),
        [
            ('font-family', MAIN_FONT_FAMILY),
            ('font-weight', MAIN_FONT_WEIGHT)
        ],
    )

    return html

if __name__ == '__main__':
    srcfilepath = Path(__file__).parent / '../../utils.py'
    with open(srcfilepath, 'r') as srcfile:
        lines = srcfile.readlines()

    signatures = []
    for line in lines:
        if len(line.split()) > 0 and line.split()[0].strip() == 'def':
            signatures.append(line.strip())

    prefix = 'utils'
    html = []
    for signature in signatures:
        html.append(funcToHTML(signature, prefix))

    destfilepath = Path(__file__).parent / 'function_signatures.md'
    with open(destfilepath, 'w') as destfile:
        for element in html:
            destfile.write(element)
            destfile.write('\n\n')