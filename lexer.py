import sys
import re
import ply.lex as lex


tokens = (
    'IF', 'DEFUN',  # Palavras-chave
    'EQ', 'NE', 'LT', 'LE', 'GT', 'GE',  # Operadores de comparação
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'INTDIV', 'MOD', 'EXP',  # Operadores aritméticos
    'LPAREN', 'RPAREN',  # Delimitadores
    'NUM', 'ID',  # Número e identificadores
)

# palavras reservadas (em minúsculas no código fonte)
reserved_map = {
    'if': 'IF',
    'defun': 'DEFUN',

    # comparação 
    'eq': 'EQ',
    'ne': 'NE',
    'lt': 'LT',
    'le': 'LE',
    'gt': 'GT',
    'ge': 'GE',

    # operadores
    'div': 'INTDIV',
    'mod': 'MOD',
    'exp': 'EXP',
}

# Ignorar espaços e tabs
t_ignore = ' \t\r'



def t_comment_block(t):
    r'/\*([^*]|\*+[^*/])*\*+/'
    pass


def t_comment_line(t):  # comentários // até o fim da linha
    r'//[^\n]*'
    pass


#comentário de linha estilo Lisp: ";" até o fim da linha
def t_comment_line_lisp(t):
    r';[^\n]*'
    pass


#quebra de linha (contagem) com o lineno
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


#delimitadores
t_LPAREN = r'\('
t_RPAREN = r'\)'


#operadores de comparação (ordem importa: mais longos primeiro)
t_LE = r'<='
t_GE = r'>='
t_NE = r'!='
t_EQ = r'==|='  
t_LT = r'<'
t_GT = r'>'


#operadores aritméticos simbólicos
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'


#número (inteiro/float com expoente)
number_regex = r'\d+(\.\d+)?([eE][+-]?\d+)?'


@lex.TOKEN(number_regex)
def t_NUM(t):
    text = t.value
    # tenta converter para int se não tiver ponto nem expoente
    if re.match(r'^\d+$', text):
        t.value = int(text)
    else:
        t.value = float(text)
    return t


#Identificadores e palavras reservadas
id_regex = r'[A-Za-z_][A-Za-z0-9_]*'  # nomes que começam com letra/underline e seguem com letras/dígitos/underline


@lex.TOKEN(id_regex)
def t_ID(t):
    typ = reserved_map.get(t.value.lower())
    if typ:
        t.type = typ
    return t


def t_error(t):  #Mostrar a linha problemática
    col = find_column(t.lexer.lexdata, t)
    print(f'ERROR 404! -> Caracter inesperado "{t.value[0]}" na linha {t.lexer.lineno}, coluna {col}')
    t.lexer.skip(1)



def find_column(text, token):
   
    last_cr = text.rfind('\n', 0, token.lexpos)
    #converte posição absoluta  em número de coluna na linha
    if last_cr < 0:
        last_cr = -1
    return token.lexpos - last_cr


#etorna lista de tokens 
def scan(text):
    lexer = lex.lex()
    lexer.input(text)
    out = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        # Formato: <TIPO, lexema, linha, coluna>
        col = find_column(lexer.lexdata, tok)
        out.append((tok.type, tok.value, tok.lineno, col))
    return out


def main():
    # Se um arquivo for passado por argumento, lê dele, senão, usa o exemplo
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            src = f.read()
    else:
        # Exemplo LISP + comentários suportados
        src = """
        ; comentário estilo Lisp
        (defun soma (a b)
          (+ a b))  // comentário //

        /* bloco de comentário
           multilinha */
        (if (le 2 3)
            (soma 10 (exp 2 3))
            (mod 7 3))

        (= 5 5)  ; igualdade simbólica
        (/ (* 3 4) 2)
        """

    tokens_list = scan(src)
    for tkn in tokens_list:
        token_type, token_lexeme, line, col = tkn
        print(f'<{token_type}, {token_lexeme}> | linha {line}, col {col}')


if __name__ == '__main__':
    main()
