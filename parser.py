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
    # devolve o nome em minúsculas, ex.: 'lt', 'eq', ...
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


# (árvore bonita)

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


# ---------- GERADOR DE CÓDIGO INTERMEDIÁRIO (QUÁDRUPLAS) ----------

class CodeGenerator(object):
    def __init__(self):
        # vetor de instruções: cada uma é (op, arg1, arg2, result)
        self.code = []
        self.temp_count = 0
        self.label_count = 0
        # tabela de funções: nome -> {"label": label_entrada, "params": [...]}
        self.functions = {}

    # --- utilitários de geração ---

    def new_temp(self):
        self.temp_count += 1
        return f"t{self.temp_count}"

    def new_label(self, prefix="L"):
        self.label_count += 1
        return f"{prefix}{self.label_count}"

    def emit(self, op, arg1=None, arg2=None, result=None):
        quad = (op, arg1, arg2, result)
        self.code.append(quad)
        return len(self.code) - 1

    # --- geração de código a partir da AST ---

    def gen_program(self, ast_list):
        """Recebe a AST retornada por parse_code (lista de forms de topo)
        e gera o vetor de quádruplas. Retorna o índice de início do "main".
        """
        # 1) Gera código para as funções (defun)
        for node in ast_list:
            if isinstance(node, tuple) and node and node[0] == 'def':
                self._gen_def(node)

        # marca onde começa o código "principal"
        start_main = len(self.code)

        # 2) Gera código para os forms de topo que NÃO são defun
        for node in ast_list:
            if isinstance(node, tuple) and node and node[0] == 'def':
                continue
            result_temp = self.gen_expr(node)
            # para efeito de teste/demonstração: imprime cada resultado de topo
            self.emit('print', result_temp, None, None)

        return start_main

    def _gen_def(self, node):
        """node = ('def', name, params, body)"""
        _, name, params, body = node
        label_name = f"func_{name}"

        # registra a função
        self.functions[name] = {
            "label": label_name,
            "params": params,
        }

        # rótulo de entrada da função
        self.emit('label', None, None, label_name)

        # gera código do corpo
        ret_temp = self.gen_expr(body)

        # valor de retorno = valor da última expressão
        self.emit('ret', ret_temp, None, None)

    def gen_expr(self, node):
        """Gera código para uma expressão e devolve o nome do temporário
        (ou variável) que contém o resultado.
        """

        # Caso caia uma lista (sequência de forms), usa o valor do último
        if isinstance(node, list):
            last = None
            for n in node:
                last = self.gen_expr(n)
            return last

        # Literal solto (não muito comum)
        if not isinstance(node, tuple) or not node:
            t = self.new_temp()
            self.emit('const', node, None, t)
            return t

        tag = node[0]

        # ('num', valor)
        if tag == 'num':
            value = node[1]
            t = self.new_temp()
            self.emit('const', value, None, t)
            return t

        # ('id', nome)
        if tag == 'id':
            # não gera instrução; referencia direta ao nome
            return node[1]

        # ('binop', op, left, right)
        if tag == 'binop':
            _, op, left, right = node
            t1 = self.gen_expr(left)
            t2 = self.gen_expr(right)
            res = self.new_temp()
            # op ∈ {+, -, *, /, div, mod, exp}
            self.emit(op, t1, t2, res)
            return res

        # ('cmp', op, left, right)
        if tag == 'cmp':
            _, op, left, right = node
            t1 = self.gen_expr(left)
            t2 = self.gen_expr(right)
            res = self.new_temp()
            # op ∈ {lt, le, gt, ge, eq, ne}
            self.emit(op, t1, t2, res)
            return res

        # ('if', cond, then_expr, else_expr)
        if tag == 'if':
            _, cond, then_expr, else_expr = node
            cond_temp = self.gen_expr(cond)
            else_label = self.new_label("else")
            end_label = self.new_label("endif")
            res = self.new_temp()

            # se cond == false → pula para else
            self.emit('if_false', cond_temp, None, else_label)

            # then
            t_then = self.gen_expr(then_expr)
            self.emit('mov', t_then, None, res)
            self.emit('goto', None, None, end_label)

            # else
            self.emit('label', None, None, else_label)
            t_else = self.gen_expr(else_expr)
            self.emit('mov', t_else, None, res)

            # fim do if
            self.emit('label', None, None, end_label)
            return res

        # ('call', name, args)
        if tag == 'call':
            _, name, args = node
            arg_temps = []
            for arg in args:
                arg_temps.append(self.gen_expr(arg))

            # gera PARAM para cada argumento
            for t in arg_temps:
                self.emit('param', t, None, None)

            # gera CALL que devolve um temporário
            res = self.new_temp()
            self.emit('call', name, len(args), res)
            return res

        if tag == 'def':
            # defun só deve aparecer no topo, não como expressão interna
            raise ValueError("Nó 'def' não deve aparecer dentro de expressão.")

        raise ValueError(f"Tag de nó desconhecida na AST: {tag!r}")

    # --- utilitário: dump em arquivo de texto ---

    def dump_to_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for i, (op, a1, a2, res) in enumerate(self.code):
                f.write(f"{i:03}: ({op}, {a1}, {a2}, {res})\n")


# ---------- AMBIENTE DE EXECUÇÃO (INTERPRETADOR DE QUÁDRUPLAS) ----------

class Frame(object):
    def __init__(self, name, env, ip, ret_ip=None, ret_var=None):
        self.name = name      # nome da função (ou "<main>")
        self.env = env        # dicionário: nome -> valor
        self.ip = ip          # índice da próxima instrução
        self.ret_ip = ret_ip  # índice de retorno no chamador
        self.ret_var = ret_var  # em qual variável do chamador guardar o retorno


def run_intermediate(code, functions, start_ip=0, debug=False):
    """Interpreta o vetor de quádruplas gerado pelo CodeGenerator.

    code      : lista de quádruplas (op, arg1, arg2, result)
    functions : tabela de funções {nome: {"label": label, "params": [...]}}
    start_ip  : índice de início do "main" no vetor de código
    """

    # 1) constrói mapa de rótulos: nome_label -> índice no vetor
    label_pos = {}
    for i, (op, a1, a2, res) in enumerate(code):
        if op == 'label':
            label_pos[res] = i

    def get_val(frame, x):
        # obtém valor de um operando (pode ser constante, nome de var ou None)
        if isinstance(x, (int, float, bool)):
            return x
        if x is None:
            return None
        return frame.env.get(x, 0)

    # pilha de frames de execução (para chamadas de função)
    call_stack = [Frame("<main>", {}, start_ip)]
    # pilha de parâmetros (usada pelos PARAM/CALL)
    param_stack = []
    # para guardar valores impressos (útil em testes)
    outputs = []

    while call_stack:
        frame = call_stack[-1]

        # se ip sair do código, termina esse frame
        if frame.ip < 0 or frame.ip >= len(code):
            call_stack.pop()
            continue

        op, a1, a2, res = code[frame.ip]

        if debug:
            print(f"[{frame.name}] ip={frame.ip:03}  {(op, a1, a2, res)}  env={frame.env}")

        # ----- implementação de cada opcode -----

        if op == 'const':
            frame.env[res] = a1
            frame.ip += 1

        elif op in ('+', '-', '*', '/', 'div', 'mod', 'exp',
                    'lt', 'le', 'gt', 'ge', 'eq', 'ne'):
            v1 = get_val(frame, a1)
            v2 = get_val(frame, a2)
            if op == '+':
                r = v1 + v2
            elif op == '-':
                r = v1 - v2
            elif op == '*':
                r = v1 * v2
            elif op == '/':
                r = v1 / v2
            elif op == 'div':
                r = v1 // v2
            elif op == 'mod':
                r = v1 % v2
            elif op == 'exp':
                r = v1 ** v2
            elif op == 'lt':
                r = v1 < v2
            elif op == 'le':
                r = v1 <= v2
            elif op == 'gt':
                r = v1 > v2
            elif op == 'ge':
                r = v1 >= v2
            elif op == 'eq':
                r = v1 == v2
            elif op == 'ne':
                r = v1 != v2
            frame.env[res] = r
            frame.ip += 1

        elif op == 'mov':
            frame.env[res] = get_val(frame, a1)
            frame.ip += 1

        elif op == 'label':
            # rótulo não faz nada em tempo de execução
            frame.ip += 1

        elif op == 'goto':
            frame.ip = label_pos[res]

        elif op == 'if_false':
            cond = get_val(frame, a1)
            if not cond:
                frame.ip = label_pos[res]
            else:
                frame.ip += 1

        elif op == 'param':
            val = get_val(frame, a1)
            param_stack.append(val)
            frame.ip += 1

        elif op == 'call':
            func_name = a1
            n_args = a2
            info = functions.get(func_name)
            if info is None:
                raise RuntimeError(f"Função não declarada: {func_name}")
            label = info["label"]
            params = info["params"]

            if n_args != len(params):
                raise RuntimeError(
                    f"Chamada de {func_name} com {n_args} args, "
                    f"mas a função espera {len(params)}."
                )

            # pega os últimos n_args da pilha de parâmetros
            args = param_stack[-n_args:]
            del param_stack[-n_args:]

            # cria ambiente local da função
            new_env = {}
            for pname, pval in zip(params, args):
                new_env[pname] = pval

            func_ip = label_pos[label]

            # configura retorno no frame atual (chamador)
            ret_ip = frame.ip + 1
            ret_var = res

            # empilha frame da função chamada
            call_stack.append(Frame(func_name, new_env, func_ip, ret_ip, ret_var))

        elif op == 'ret':
            ret_val = get_val(frame, a1)
            finished = call_stack.pop()
            if not call_stack:
                # terminou tudo (retorno final de "main" ou função top-level)
                outputs.append(ret_val)
                break
            caller = call_stack[-1]
            if finished.ret_var is not None:
                caller.env[finished.ret_var] = ret_val
            # volta para o ponto de retorno
            caller.ip = finished.ret_ip

        elif op == 'print':
            val = get_val(frame, a1)
            outputs.append(val)
            print(f"=> {val}")
            frame.ip += 1

        else:
            raise RuntimeError(f"Opcode desconhecido: {op}")

    return outputs


# Função utilitária de parsing

def parse_code(src, dump_productions=False):
    parser = yacc.yacc(write_tables=False)

    import ply.lex as lex
    import lexer as lexmod
    lx = lex.lex(module=lexmod)

    if dump_productions:
        with open('nome.txt', 'w', encoding='utf-8') as f:
            f.write('Regras do Parser (LISP, binário):\n\n')
            for r in parser.productions:
                f.write(str(r) + '\n')
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
        code = """(defun soma (a b)
  (+ a b))

(if (lt 2 3)
    (soma 10 (exp 2 3))   ; exp binário: 2^3
    (mod 7 3))

(/ (* 3 4) 2)   ; exemplo de binário aninhado
(= 5 5)
"""

    # 1) Parsing → AST
    ast = parse_code(code)

    # 2) Geração de código intermediário
    cg = CodeGenerator()
    start_main = cg.gen_program(ast)

    print("\nCódigo intermediário (quádruplas <op, arg1, arg2, result>):\n")
    for i, q in enumerate(cg.code):
        print(f"{i:03}: {q}")

    # 3) Salva em arquivo texto
    cg.dump_to_file("codigo_intermediario.txt")
    print('\nArquivo "codigo_intermediario.txt" gerado.\n')

    # 4) Executa no ambiente de execução
    print("Saída do ambiente de execução:\n")
    run_intermediate(cg.code, cg.functions, start_main)
