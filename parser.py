# -*- coding: utf-8 -*-
import sys
import ply.yacc as yacc
from lexer import tokens  # usa tokens do lexer.py

start = 'program'  # símbolo inicial

# Programa = 1+ formulários

def p_program(p):
    'program : forms_plus'
    p[0] = p[1]  # lista de ASTs (um por formulário de topo)

def p_forms_plus(p):
    '''forms_plus : forms_plus form
                  | form'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


# Formas básicas

def p_form_num(p):
    'form : NUM'
    p[0] = ('num', p[1])

def p_form_id(p):
    'form : ID'
    p[0] = ('id', p[1])

# (if cond then else)
def p_form_if(p):
    'form : LPAREN IF form form form RPAREN'
    p[0] = ('if', p[3], p[4], p[5])

# (defun nome (params...) body)
def p_form_defun(p):
    'form : LPAREN DEFUN ID LPAREN params_opt RPAREN form RPAREN'
    params = p[5] if p[5] is not None else []
    p[0] = ('def', p[3], params, p[7])


# Operadores prefixados e chamadas

def p_form_arith_bin(p):
    'form : LPAREN op_arith form form RPAREN'
    # agora binário apenas (nada de listas N-árias)
    p[0] = ('binop', p[2], p[3], p[4])

# Comparações binárias: (lt a b), (eq x y), ...
def p_form_cmp(p):
    'form : LPAREN op_cmp form form RPAREN'
    p[0] = ('cmp', p[2], p[3], p[4])

# Chamada genérica: (fun arg1 arg2 ...)
def p_form_call(p):
    'form : LPAREN ID forms_opt RPAREN'
    args = p[3] if p[3] is not None else []
    p[0] = ('call', p[2], args)


# Cabeças (operadores)

def p_op_arith(p):
    '''op_arith : PLUS
                | MINUS
                | TIMES
                | DIVIDE
                | INTDIV
                | MOD
                | EXP'''
    m = {
        'PLUS': '+', 'MINUS': '-', 'TIMES': '*', 'DIVIDE': '/',
        'INTDIV': 'div', 'MOD': 'mod', 'EXP': 'exp'
    }
    p[0] = m[p.slice[1].type]

def p_op_cmp(p):
    '''op_cmp : EQ
              | NE
              | LT
              | LE
              | GT
              | GE'''
    p[0] = p.slice[1].type.lower()


# Listas de argumentos e parâmetros

def p_forms_opt(p):
    '''forms_opt : forms
                 | empty'''
    p[0] = p[1]

def p_forms(p):
    '''forms : forms form
             | form'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_params_opt(p):
    '''params_opt : params
                  | empty'''
    p[0] = p[1]

def p_params(p):
    '''params : params ID
              | ID'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


# Vazio e erro

def p_empty(p):
    'empty :'
    p[0] = None

def p_error(p):
    if p:
        print(f"Erro de sintaxe próximo de {getattr(p,'value','?')} (token {getattr(p,'type','?')})")
    else:
        print("Erro de sintaxe: fim inesperado.")


# (árvore)

def _pp(node, prefix=""):
    # node pode ser uma lista de forms (topo) OU um nó (tuple)
    if isinstance(node, list):
        print(prefix + "Program")
        for i, n in enumerate(node):
            last = (i == len(node) - 1)
            branch = "└─ " if last else "├─ "
            _pp(n, prefix + branch)
        return

    if not isinstance(node, tuple) or not node:
        print(prefix + repr(node))
        return

    tag = node[0]

    if tag == 'num':
        print(prefix + f"num: {node[1]}")
    elif tag == 'id':
        print(prefix + f"id: {node[1]}")
    elif tag == 'def':
        _, name, params, body = node
        print(prefix + f"def {name} ({', '.join(params)})")
        _pp(body, prefix + "   ")
    elif tag == 'if':
        _, cond, then, els = node
        print(prefix + "if")
        _pp(cond, prefix + "   ├─ cond: ")
        _pp(then, prefix + "   ├─ then: ")
        _pp(els,  prefix + "   └─ else: ")
    elif tag == 'binop':
        _, op, left, right = node
        print(prefix + f"binop {op}")
        _pp(left,  prefix + "   ├─ ")
        _pp(right, prefix + "   └─ ")
    elif tag == 'cmp':
        _, op, left, right = node
        print(prefix + f"cmp {op}")
        _pp(left,  prefix + "   ├─ ")
        _pp(right, prefix + "   └─ ")
    elif tag == 'call':
        _, name, args = node
        print(prefix + f"call {name}")
        for i, a in enumerate(args):
            last = (i == len(args) - 1)
            branch = "   └─ " if last else "   ├─ "
            _pp(a, prefix + branch)
    else:
        print(prefix + repr(node))


# Função utilitária

def parse_code(src, dump_productions=False):
    parser = yacc.yacc(write_tables=False)

    import ply.lex as lex
    import lexer as lexmod
    lx = lex.lex(module=lexmod)

    if dump_productions:
        with open('nome.txt','w',encoding='utf-8') as f:
            f.write('Regras do Parser (LISP, binário):\n\n')
            for r in parser.productions:
                f.write(str(r)+'\n')
        print('Regras salvas em nome.txt')

    ast = parser.parse(src, lexer=lx)
    print('AST (lista de forms):')
    print(ast)
    print('\nÁrvore:\n')
    _pp(ast)
    return ast


# Execução direta (teste)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            code = f.read()
    else:
        code = """
(defun soma (a b)
  (+ a b))

(if (lt 2 3)
    (soma 10 (exp 3))   ; exp agora binário? então, por exemplo, (exp 2 3) para 2^3
    (mod 7 3))

(/ (* 3 4) 2)   ; exemplo de binário aninhado (no estilo que você pediu)
(= 5 5)
"""
    parse_code(code)
