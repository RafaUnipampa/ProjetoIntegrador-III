import sys
import ply.yacc as yacc
from lexer import tokens  # usa tokens do lexer.py

start = 'program'

# Regras da gramática
def p_program(p):
    'program : forms_plus'
    p[0] = p[1]

def p_forms_plus(p):
    """forms_plus : forms_plus form
                  | form"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_form_num(p):
    'form : NUM'
    p[0] = ('num', p[1])

def p_form_id(p):
    'form : ID'
    p[0] = ('id', p[1])

def p_form_if(p):
    'form : LPAREN IF form form form RPAREN'
    p[0] = ('if', p[3], p[4], p[5])

def p_form_defun(p):
    'form : LPAREN DEFUN ID LPAREN params_opt RPAREN form RPAREN'
    params = p[5] if p[5] is not None else []
    p[0] = ('def', p[3], params, p[7])

def p_form_arith_bin(p):
    'form : LPAREN op_arith form form RPAREN'
    p[0] = ('binop', p[2], p[3], p[4])

def p_form_cmp(p):
    'form : LPAREN op_cmp form form RPAREN'
    p[0] = ('cmp', p[2], p[3], p[4])

def p_form_call(p):
    'form : LPAREN ID forms_opt RPAREN'
    args = p[3] if p[3] is not None else []
    p[0] = ('call', p[2], args)

def p_op_arith(p):
    """op_arith : PLUS
                | MINUS
                | TIMES
                | DIVIDE
                | INTDIV
                | MOD
                | EXP"""
    m = {
        'PLUS': '+', 'MINUS': '-', 'TIMES': '*', 'DIVIDE': '/',
        'INTDIV': 'div', 'MOD': 'mod', 'EXP': 'exp'
    }
    p[0] = m[p.slice[1].type]

def p_op_cmp(p):
    """op_cmp : EQ
              | NE
              | LT
              | LE
              | GT
              | GE"""
    p[0] = p.slice[1].type.lower()

def p_forms_opt(p):
    """forms_opt : forms
                 | empty"""
    p[0] = p[1]

def p_forms(p):
    """forms : forms form
             | form"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_params_opt(p):
    """params_opt : params
                  | empty"""
    p[0] = p[1]

def p_params(p):
    """params : params ID
              | ID"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_empty(p):
    'empty :'
    p[0] = None

def p_error(p):
    if p:
        print(f"Erro de sintaxe proximo de {getattr(p,'value','?')} (token {getattr(p,'type','?')})")
    else:
        print("Erro de sintaxe: fim inesperado.")

# Impressão da AST (opcional, para debug)
def _pp(node, prefix=""):
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

# Gerador de código intermediário (quádruplas)
class CodeGenerator(object):
    def __init__(self):
        # vetor de instruções: cada uma é (op, arg1, arg2, result)
        self.code = []
        self.data = []                # # NOVO: área de dados para variáveis e constantes
        self.const_table = {}         # # NOVO: mapa de constantes -> endereço na área de dados
        self.functions = {}           # # ALTERADO: tabela de funções: nome -> {"addr": ..., "param_count": ..., "frame_size": ...}
        self.label_count = 0
        # controle de contexto de geração
        self.in_function = False      # # NOVO: indica se estamos gerando dentro de uma função (incluindo main)
        self.cur_param_map = None     # # NOVO: mapeamento de parâmetros (nome -> endereço)
        self.cur_offset_count = 0     # # NOVO: contador de endereços locais usados no frame atual
        self.main_frame_size = 0      # # NOVO: tamanho do frame do código principal (main)
        # mapa de opcode para código numérico
        self.opcodes = {              # # NOVO: mapeamento de operações para códigos inteiros
            'const': 1, 'mov': 2, 'goto': 3, 'if_false': 4,
            'param': 5, 'call': 6, 'ret': 7, 'print': 8, 'label': 9,
            '+': 10, '-': 11, '*': 12, '/': 13, 'div': 14, 'mod': 15, 'exp': 16,
            'lt': 17, 'le': 18, 'gt': 19, 'ge': 20, 'eq': 21, 'ne': 22
        }
        # mapa inverso (opcional, para debug/impressão)
        self.rev_opcodes = {v: k for k, v in self.opcodes.items()}

    def new_label(self, prefix="L"):
        self.label_count += 1
        return self.label_count         # # ALTERADO: retorna um ID numérico de rótulo em vez de string

    def emit(self, op, arg1=None, arg2=None, result=None):
        op_code = self.opcodes.get(op, op)  # # ALTERADO: converte operação para código numérico
        quad = (op_code, arg1, arg2, result)
        self.code.append(quad)
        return len(self.code) - 1

    def gen_program(self, ast_list):
        """Gera o vetor de quádruplas para o programa completo. Retorna índice de início do main."""
        # 1) Gera código para funções definidas (defun)
        for node in ast_list:
            if isinstance(node, tuple) and node and node[0] == 'def':
                self._gen_def(node)
        # 2) Código "main" começa aqui
        start_main = len(self.code)
        # prepara contexto para main (sem parâmetros)
        self.in_function = True        # # NOVO
        self.cur_param_map = {}        # # NOVO
        self.cur_offset_count = 0      # # NOVO
        # Gera código para forms de topo (que não são defun)
        for node in ast_list:
            if isinstance(node, tuple) and node and node[0] == 'def':
                continue
            result_addr = self.gen_expr(node)
            # imprime cada resultado de topo
            self.emit('print', result_addr, None, None)
        # tamanho do frame do main = quantidade de temporários utilizados
        self.main_frame_size = self.cur_offset_count  # # NOVO
        return start_main

    def _gen_def(self, node):
        """Gera código para uma definição de função (defun)."""
        _, name, params, body = node
        # contexto de função
        self.in_function = True                     # # NOVO
        self.cur_param_map = {}
        # atribui endereços (offsets negativos) para parâmetros
        for i, p_name in enumerate(params):
            self.cur_param_map[p_name] = -(i + 1)   # # NOVO: param -> -1, -2, ...
        self.cur_offset_count = len(params)         # # NOVO: contador inicia no número de parâmetros
        # rótulo de entrada da função
        func_label_id = self.new_label("func")      # # NOVO: id numérico para rótulo da função
        func_start_index = self.emit('label', None, None, func_label_id)
        # registra função
        self.functions[name] = {                    # # ALTERADO: guarda endereço e assinatura da função
            "addr": func_start_index,
            "param_count": len(params),
            "frame_size": None  # será preenchido após gerar o corpo
        }
        # gera código do corpo da função
        ret_addr = self.gen_expr(body)
        # emite instrução de retorno (valor de retorno = ret_addr)
        self.emit('ret', ret_addr, None, None)
        # define tamanho do frame da função (params + temporários locais)
        self.functions[name]['frame_size'] = self.cur_offset_count  # # NOVO
        # restaura contexto
        self.in_function = False
        self.cur_param_map = None
        self.cur_offset_count = 0

    def gen_expr(self, node):
        """Gera código para uma expressão e retorna o endereço do resultado na área de dados/stack."""
        # lista de forms: avalia sequencialmente e retorna valor do último
        if isinstance(node, list):
            last_addr = None
            for n in node:
                last_addr = self.gen_expr(n)
            return last_addr
        # literal atômico (número, booleano, etc)
        if not isinstance(node, tuple) or not node:
            const_val = node
            # insere constante na área de dados se ainda não está lá
            if const_val not in self.const_table:
                addr = len(self.data)
                self.const_table[const_val] = addr
                self.data.append(const_val)        # # NOVO: constante armazenada na área de dados
            const_addr = self.const_table[const_val]
            # aloca temporário para armazenar o valor da constante
            res_addr = self.new_temp()
            self.emit('const', const_addr, None, res_addr)
            return res_addr
        tag = node[0]
        if tag == 'num':
            value = node[1]
            if value not in self.const_table:
                addr = len(self.data)
                self.const_table[value] = addr
                self.data.append(value)            # # NOVO
            const_addr = self.const_table[value]
            res_addr = self.new_temp()
            self.emit('const', const_addr, None, res_addr)
            return res_addr
        if tag == 'id':
            name = node[1]
            # retorna endereço associado ao identificador (parâmetro ou variável)
            if self.cur_param_map and name in self.cur_param_map:
                return self.cur_param_map[name]    # # ALTERADO: retorna endereço (ponteiro) ao invés de nome
            else:
                # identificador não encontrado (fora de contexto): retorna endereço 0 (valor default)
                return 0                           # # NOVO
        if tag == 'binop':
            _, op, left, right = node
            addr1 = self.gen_expr(left)
            addr2 = self.gen_expr(right)
            res_addr = self.new_temp()
            self.emit(op, addr1, addr2, res_addr)
            return res_addr
        if tag == 'cmp':
            _, op, left, right = node
            addr1 = self.gen_expr(left)
            addr2 = self.gen_expr(right)
            res_addr = self.new_temp()
            self.emit(op, addr1, addr2, res_addr)
            return res_addr
        if tag == 'if':
            _, cond, then_expr, else_expr = node
            cond_addr = self.gen_expr(cond)
            else_label = self.new_label("else")
            end_label = self.new_label("endif")
            res_addr = self.new_temp()
            # se condição falsa -> pular para else
            self.emit('if_false', cond_addr, None, else_label)
            # then
            then_val = self.gen_expr(then_expr)
            self.emit('mov', then_val, None, res_addr)
            self.emit('goto', None, None, end_label)
            # else
            self.emit('label', None, None, else_label)
            else_val = self.gen_expr(else_expr)
            self.emit('mov', else_val, None, res_addr)
            # fim do if
            self.emit('label', None, None, end_label)
            return res_addr
        if tag == 'call':
            _, name, args = node
            # avalia argumentos
            arg_addrs = []
            for arg in args:
                arg_addrs.append(self.gen_expr(arg))
            # empilha parâmetros
            for addr in arg_addrs:
                self.emit('param', addr, None, None)
            # chamada de função
            res_addr = self.new_temp()
            # obtém endereço da função alvo
            func_info = self.functions.get(name)
            if func_info:
                func_addr = func_info['addr']
            else:
                func_addr = -1  # função não definida (erro será lançado em tempo de execução) # NOVO
            self.emit('call', func_addr, len(args), res_addr)
            return res_addr
        if tag == 'def':
            raise ValueError("Nó 'def' não deve aparecer dentro de expressão.")
        raise ValueError(f"Tag de nó desconhecida na AST: {tag!r}")

    def new_temp(self):
        """Aloca um novo temporário e retorna seu endereço (inteiro)."""
        if self.in_function:
            addr = -(self.cur_offset_count + 1)
            self.cur_offset_count += 1
            return addr              #A endereço de temporário local (offset negativo)
        else:
            addr = len(self.data)
            self.data.append(0)      #N reserva espaço na área de dados para temporário global (main)
            return addr

    def dump_to_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for i, (op, a1, a2, res) in enumerate(self.code):
                op_name = self.rev_opcodes.get(op, op)
                f.write(f"{i:03}: ({op_name}, {a1}, {a2}, {res})\n")

# Ambiente de execução (interpretador)
def run_intermediate(code, functions, start_ip=0, data=None, main_frame_size=0, debug=False):
    """Interpreta o vetor de quádruplas gerado pelo CodeGenerator, usando áreas de memória."""
    # inicializa áreas de memória
    data_mem = data[:] if data is not None else []   #N área de dados (variáveis globais/constantes)
    stack_mem = []                                   #N área de stack (pilha de execução)
    BP = 0                                           #N base pointer (início do frame atual na pilha)
    SP = 0                                           #N stack pointer (próxima posição livre na pilha)
    IP = start_ip                                    #próxima instrução a executar
    # aloca espaço para frame do main na pilha
    for i in range(main_frame_size):
        stack_mem.append(0)
    SP += main_frame_size                            # # NOVO: reserva espaço na stack para variáveis do main
    param_buffer = []                                # # ALTERADO: pilha de parâmetros (lista de valores)
    outputs = []
    # mapeia tamanho do frame por base pointer (para limpeza ao retornar)
    frame_size_map = {BP: main_frame_size}           # # NOVO: registra tamanho do frame principal

    def get_value(addr):
        # obtém valor a partir de um endereço (na área de dados ou stack)
        if addr is None:
            return None
        if isinstance(addr, bool):
            # bool usado diretamente
            return addr
        if addr >= 0:
            return data_mem[addr] if addr < len(data_mem) else 0
        else:
            idx = BP + (-addr - 1)
            return stack_mem[idx] if 0 <= idx < len(stack_mem) else 0

    def set_value(addr, value):
        # escreve um valor em um endereço (na área de dados ou stack)
        if addr is None:
            return
        if addr >= 0:
            if addr >= len(data_mem):
                data_mem.extend([0] * (addr - len(data_mem) + 1))
            data_mem[addr] = value
        else:
            idx = BP + (-addr - 1)
            if idx >= len(stack_mem):
                stack_mem.extend([0] * (idx - len(stack_mem) + 1))
            stack_mem[idx] = value

    # loop de execução das instruções
    while True:
        if IP < 0 or IP >= len(code):
            break  # fim da execução
        op, a1, a2, res = code[IP]
        if debug:
            print(f"[IP={IP:03}] (op={op}, a1={a1}, a2={a2}, res={res})  BP={BP} SP={SP}")  # # NOVO: saída de debug
        # verifica o opcode e executa a ação correspondente
        if op == 1:  # const
            # copia valor constante da área de dados para o destino
            val = get_value(a1)
            set_value(res, val)
            IP += 1
        elif op in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22):
            # operações aritméticas e comparações
            v1 = get_value(a1)
            v2 = get_value(a2)
            if op == 10:    # +
                result_val = v1 + v2
            elif op == 11:  # -
                result_val = v1 - v2
            elif op == 12:  # *
                result_val = v1 * v2
            elif op == 13:  # /
                result_val = v1 / v2
            elif op == 14:  # div (divisão inteira)
                result_val = v1 // v2
            elif op == 15:  # mod
                result_val = v1 % v2
            elif op == 16:  # exp
                result_val = v1 ** v2
            elif op == 17:  # lt
                result_val = (v1 < v2)
            elif op == 18:  # le
                result_val = (v1 <= v2)
            elif op == 19:  # gt
                result_val = (v1 > v2)
            elif op == 20:  # ge
                result_val = (v1 >= v2)
            elif op == 21:  # eq
                result_val = (v1 == v2)
            elif op == 22:  # ne
                result_val = (v1 != v2)
            else:
                result_val = 0
            set_value(res, result_val)
            IP += 1
        elif op == 2:  # mov
            val = get_value(a1)
            set_value(res, val)
            IP += 1
        elif op == 9:  # label
            # rótulo: nenhuma ação necessária em tempo de execução
            IP += 1
        elif op == 3:  # goto
            IP = res  # desvia para a instrução no índice res
        elif op == 4:  # if_false
            cond_val = get_value(a1)
            if not cond_val:
                IP = res  # se condição falsa, pula para o rótulo (índice) indicado em res
            else:
                IP += 1
        elif op == 5:  # param
            val = get_value(a1)
            param_buffer.append(val)
            IP += 1
        elif op == 6:  # call
            func_addr = a1   # endereço de início da função chamada
            n_args = a2      # número de argumentos
            # obtém informação da função alvo pelo endereço
            target_func = None
            for fname, finfo in functions.items():
                if isinstance(finfo, dict) and finfo.get("addr") == func_addr:
                    target_func = finfo
                    break
            if target_func is None or func_addr < 0:
                raise RuntimeError(f"Função não declarada: {a1}")
            if n_args != target_func['param_count']:
                raise RuntimeError(f"Chamada de função com {n_args} args, mas {target_func['param_count']} esperados.")
            # obtém valores dos argumentos do buffer de parâmetros
            args = param_buffer[-n_args:] if n_args > 0 else []
            if n_args > 0:
                del param_buffer[-n_args:]
            # empilha endereço de retorno, base pointer atual e destino do retorno
            ret_ip = IP + 1
            old_BP = BP
            ret_dest = res
            stack_mem.append(ret_ip); SP += 1
            stack_mem.append(old_BP); SP += 1
            stack_mem.append(ret_dest); SP += 1
            # configura novo frame na pilha: define nova base e insere argumentos
            BP = SP
            for val in args:
                stack_mem.append(val); SP += 1
            # aloca espaço para variáveis locais (frame_size inclui parâmetros e locais)
            frame_total = target_func['frame_size']
            local_count = frame_total - n_args
            for i in range(local_count):
                stack_mem.append(0); SP += 1
            # registra tamanho do frame atual
            frame_size_map[BP] = frame_total
            # desvia execução para o início da função chamada
            IP = func_addr
        elif op == 7:  # ret
            ret_val = get_value(a1)
            # recupera informações do frame atual (valores empilhados de retorno)
            ret_ip = stack_mem[BP - 3]
            prev_BP = stack_mem[BP - 2]
            ret_dest = stack_mem[BP - 1]
            # libera frame atual da pilha (restaura SP)
            SP = BP - 3
            stack_mem = stack_mem[:SP]  # remove elementos excedentes da pilha
            # escreve valor de retorno no destino adequado do frame chamador
            if ret_dest is not None:
                current_BP = BP
                BP = prev_BP
                set_value(ret_dest, ret_val)
                BP = current_BP
            # restaura base pointer e instruction pointer do chamador
            BP = prev_BP
            IP = ret_ip
            # se retornou para fora do frame principal, encerra execução
            if BP == 0 and (IP < 0 or IP >= len(code)):
                outputs.append(ret_val)
                break
        elif op == 8:  # print
            val = get_value(a1)
            outputs.append(val)
            print(f"=> {val}")
            IP += 1
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

# Execução de teste (se este módulo for executado diretamente)
if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            code = f.read()
    else:
        code = """(defun soma (a b)
   (+ a b))
(if (lt 2 3)
    (soma 10 (exp 2 3))
    (mod 7 3))
(/ (* 3 4) 2)
(= 5 5)
"""
    # 1) Parsing -> AST
    ast = parse_code(code)
    # 2) Geração de código intermediário
    cg = CodeGenerator()
    start_main = cg.gen_program(ast)
    print("\nCódigo intermediário (quádruplas <op, arg1, arg2, result>):\n")
    for i, quad in enumerate(cg.code):
        op, a1, a2, res = quad
        op_name = cg.rev_opcodes.get(op, op)
        print(f"{i:03}: ({op_name}, {a1}, {a2}, {res})")  # # ALTERADO: imprime nome do opcode para legibilidade
    cg.dump_to_file("codigo_intermediario.txt")
    print('\nArquivo "codigo_intermediario.txt" gerado.\n')
    # 4) Executa no ambiente de execução
    print("Saída do ambiente de execução:\n")
    run_intermediate(cg.code, cg.functions, start_main, data=cg.data, main_frame_size=cg.main_frame_size)
