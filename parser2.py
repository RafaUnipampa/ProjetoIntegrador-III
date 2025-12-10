import sys
import ply.yacc as yacc
from lexer import tokens  # usa tokens definidos em lexer.py


#  PARSER (PLY) – GERA AST EM FORMA DE ÁRVORE DE TUPLAS


start = 'program'  # símbolo inicial

# programa é uma ou mais forms
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
    # (if cond then else)
    p[0] = ('if', p[3], p[4], p[5])

def p_form_defun(p):
    'form : LPAREN DEFUN ID LPAREN params_opt RPAREN form RPAREN'
    params = p[5] if p[5] is not None else []
    p[0] = ('def', p[3], params, p[7])

def p_form_arith_bin(p):
    'form : LPAREN op_arith form form RPAREN'
    # (op e1 e2)  -> ('binop', '+', e1, e2)
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
    # converte token para string da operação
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
    # usa o nome do token em minúsculo: 'lt', 'ge', etc.
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
        print(f"Erro de sintaxe próximo de {getattr(p, 'value', '?')} (token {getattr(p, 'type', '?')})")
    else:
        print("Erro de sintaxe: fim inesperado.")


#  IMPRESSÃO BONITA DA AST (DEBUG)


def _pp(node, prefix=""):
    """Imprime a AST de forma hierárquica (somente para debug)."""
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


#  GERADOR DE CÓDIGO INTERMEDIÁRIO (QUÁDRUPLAS)


class CodeGenerator(object):
    def __init__(self):
        # vetor de instruções: cada uma é (op, arg1, arg2, result)
        self.code = []
        self.data = []                # área de dados (constantes / temporários globais)
        self.const_table = {}         # valor -> índice em self.data
        self.functions = {}           # nome -> {"addr": ..., "param_count": ..., "frame_size": ...}
        self.label_count = 0

        # contexto de geração
        self.in_function = False      # estamos gerando dentro de função?
        self.cur_param_map = None     # nome do parâmetro -> endereço (offset negativo)
        self.cur_offset_count = 0     # quantos temporários/locais já aloquei
        self.main_frame_size = 0      # tamanho do frame do "main" (top-level)

        # mapeamento de operação simbólica -> código inteiro
        self.opcodes = {
            'const': 1, 'mov': 2, 'goto': 3, 'if_false': 4,
            'param': 5, 'call': 6, 'ret': 7, 'print': 8, 'label': 9,
            '+': 10, '-': 11, '*': 12, '/': 13, 'div': 14, 'mod': 15, 'exp': 16,
            'lt': 17, 'le': 18, 'gt': 19, 'ge': 20, 'eq': 21, 'ne': 22
        }
        # mapa inverso (para debug / impressão)
        self.rev_opcodes = {v: k for k, v in self.opcodes.items()}

    def new_label(self, prefix="L"):
        """Gera um novo identificador numérico de rótulo (usado só como marcação)."""
        self.label_count += 1
        return self.label_count

    def emit(self, op, arg1=None, arg2=None, result=None):
        """Insere uma nova quádrupla no vetor de código."""
        op_code = self.opcodes.get(op, op)  # converte nome em código numérico
        quad = (op_code, arg1, arg2, result)
        self.code.append(quad)
        return len(self.code) - 1  # índice da instrução

    # ----------------- GERAÇÃO DO PROGRAMA -----------------

    def gen_program(self, ast_list):
        """
        Gera quádruplas para o programa todo.
        Retorna o índice de início da "main" (top-level).
        """
        # 1) Gera código das funções (defun)
        for node in ast_list:
            if isinstance(node, tuple) and node and node[0] == 'def':
                self._gen_def(node)

        # 2) Gera código do "main" (forms de topo que não são def)
        start_main = len(self.code)
        self.in_function = True
        self.cur_param_map = {}
        self.cur_offset_count = 0

        for node in ast_list:
            if isinstance(node, tuple) and node and node[0] == 'def':
                continue
            result_addr = self.gen_expr(node)
            # imprime o resultado de cada expressão de topo
            self.emit('print', result_addr, None, None)

        # tamanho do frame do main = quantidade de temporários utilizados
        self.main_frame_size = self.cur_offset_count
        return start_main

    def _gen_def(self, node):
        """
        Gera quádruplas para uma função (defun).
        node = ('def', nome, [params], corpo)
        """
        _, name, params, body = node

        # entra em contexto de função
        self.in_function = True
        self.cur_param_map = {}

        # parâmetros ficam com offsets negativos: -1, -2, ...
        for i, p_name in enumerate(params):
            self.cur_param_map[p_name] = -(i + 1)

        self.cur_offset_count = len(params)

        # rótulo de entrada da função (apenas marca no código)
        func_label_id = self.new_label("func")
        func_start_index = self.emit('label', None, None, func_label_id)

        # registra função
        self.functions[name] = {
            "addr": func_start_index,
            "param_count": len(params),
            "frame_size": None  # preenchido depois
        }

        # corpo da função
        ret_addr = self.gen_expr(body)
        # retorno da função
        self.emit('ret', ret_addr, None, None)

        # frame_size = parâmetros + temporários locais
        self.functions[name]['frame_size'] = self.cur_offset_count

        # sai do contexto
        self.in_function = False
        self.cur_param_map = None
        self.cur_offset_count = 0

    # ----------------- EXPRESSÕES -----------------

    def gen_expr(self, node):
        """
        Gera código para uma expressão e devolve o endereço
        onde o resultado foi armazenado (na pilha ou em data).
        """
        # lista de forms: avalia em sequência e devolve o valor do último
        if isinstance(node, list):
            last_addr = None
            for n in node:
                last_addr = self.gen_expr(n)
            return last_addr

        # literal "cru" (se aparecer) – tratamos como constante
        if not isinstance(node, tuple) or not node:
            const_val = node
            if const_val not in self.const_table:
                addr = len(self.data)
                self.const_table[const_val] = addr
                self.data.append(const_val)
            const_addr = self.const_table[const_val]
            res_addr = self.new_temp()
            self.emit('const', const_addr, None, res_addr)
            return res_addr

        tag = node[0]

        # número
        if tag == 'num':
            value = node[1]
            if value not in self.const_table:
                addr = len(self.data)
                self.const_table[value] = addr
                self.data.append(value)
            const_addr = self.const_table[value]
            res_addr = self.new_temp()
            self.emit('const', const_addr, None, res_addr)
            return res_addr

        # identificador
        if tag == 'id':
            name = node[1]
            # se for parâmetro/variável local, retorna o endereço (offset negativo)
            if self.cur_param_map and name in self.cur_param_map:
                return self.cur_param_map[name]
            # se não encontrar, por enquanto devolve 0 (poderia ser erro)
            return 0

        # operação aritmética binária
        if tag == 'binop':
            _, op, left, right = node
            addr1 = self.gen_expr(left)
            addr2 = self.gen_expr(right)
            res_addr = self.new_temp()
            self.emit(op, addr1, addr2, res_addr)
            return res_addr

        # comparação
        if tag == 'cmp':
            _, op, left, right = node
            addr1 = self.gen_expr(left)
            addr2 = self.gen_expr(right)
            res_addr = self.new_temp()
            self.emit(op, addr1, addr2, res_addr)
            return res_addr

        # if – agora usando índices de instrução (sem rótulos que causavam loop)
        if tag == 'if':
            _, cond, then_expr, else_expr = node

            # avalia condição
            cond_addr = self.gen_expr(cond)
            res_addr = self.new_temp()

            # reserva instrução de desvio condicional (alvo ainda None)
            idx_if_false = self.emit('if_false', cond_addr, None, None)

            # --- ramo then ---
            then_val = self.gen_expr(then_expr)
            self.emit('mov', then_val, None, res_addr)

            # reserva um goto para pular o else
            idx_goto_end = self.emit('goto', None, None, None)

            # --- ramo else começa aqui ---
            else_start = len(self.code)

            # ajusta a instrução de if_false para saltar para o início do else
            op, a1, a2, _ = self.code[idx_if_false]
            self.code[idx_if_false] = (op, a1, a2, else_start)

            # gera o else
            else_val = self.gen_expr(else_expr)
            self.emit('mov', else_val, None, res_addr)

            # fim do if: ponto para onde o goto deve saltar
            end_index = len(self.code)
            op, a1, a2, _ = self.code[idx_goto_end]
            self.code[idx_goto_end] = (op, a1, a2, end_index)

            return res_addr

        # chamada de função
        if tag == 'call':
            _, name, args = node
            # avalia argumentos
            arg_addrs = []
            for arg in args:
                arg_addrs.append(self.gen_expr(arg))
            # empilha parâmetros
            for addr in arg_addrs:
                self.emit('param', addr, None, None)
            # resultado da função
            res_addr = self.new_temp()
            # pega info da função
            func_info = self.functions.get(name)
            if func_info:
                func_addr = func_info['addr']
            else:
                func_addr = -1  # função não definida (erro em tempo de execução)
            self.emit('call', func_addr, len(args), res_addr)
            return res_addr

        if tag == 'def':
            raise ValueError("Nó 'def' não deve aparecer dentro de expressão.")

        raise ValueError(f"Tag de nó desconhecida na AST: {tag!r}")

    def new_temp(self):
        """
        Aloca um novo temporário e devolve seu endereço.
        Em função: offsets negativos na pilha; no main: posição em self.data.
        """
        if self.in_function:
            addr = -(self.cur_offset_count + 1)
            self.cur_offset_count += 1
            return addr
        else:
            addr = len(self.data)
            self.data.append(0)  # reserva espaço na área de dados
            return addr

    def dump_to_file(self, filename):
        """Salva as quádruplas em um arquivo de texto (para debug)."""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, (op, a1, a2, res) in enumerate(self.code):
                op_name = self.rev_opcodes.get(op, op)
                f.write(f"{i:03}: ({op_name}, {a1}, {a2}, {res})\n")


#  INTERPRETADOR DAS QUÁDRUPLAS


def run_intermediate(code, functions, start_ip=0, data=None, main_frame_size=0, debug=False):
    """
    Executa o vetor de quádruplas gerado pelo CodeGenerator.
    Usa duas memórias:
      - data_mem: globais/constantes
      - stack_mem: pilha de frames (funções / main)
    """
    # área de dados (cópia, para não alterar o original)
    data_mem = data[:] if data is not None else []
    stack_mem = []

    BP = 0   # base pointer (início do frame atual)
    SP = 0   # stack pointer (próxima posição livre)
    IP = start_ip  # instruction pointer (índice da próxima instrução)

    # reserva espaço para o frame do main na pilha
    for _ in range(main_frame_size):
        stack_mem.append(0)
    SP += main_frame_size

    param_buffer = []   # armazena valores dos parâmetros antes da chamada
    outputs = []

    # funções auxiliares para ler/escrever endereços

    def get_value(addr):
        """Lê um valor a partir de um endereço (data ou pilha)."""
        if addr is None:
            return None
        if isinstance(addr, bool):
            return addr
        if addr >= 0:
            # endereço em data_mem
            return data_mem[addr] if addr < len(data_mem) else 0
        else:
            # endereço relativo ao frame atual na pilha
            idx = BP + (-addr - 1)
            return stack_mem[idx] if 0 <= idx < len(stack_mem) else 0

    def set_value(addr, value):
        """Escreve um valor em um endereço (data ou pilha)."""
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

    # loop principal de execução
    while True:
        if IP < 0 or IP >= len(code):
            break  # acabou o programa

        op, a1, a2, res = code[IP]

        if debug:
            print(f"[IP={IP:03}] (op={op}, a1={a1}, a2={a2}, res={res})  BP={BP} SP={SP}")

        # ----------------- decodifica opcode -----------------

        if op == 1:  # const
            val = get_value(a1)   # a1 é índice da constante em data_mem
            set_value(res, val)
            IP += 1

        elif op in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22):
            # operações aritméticas e comparações
            v1 = get_value(a1)
            v2 = get_value(a2)

            if op == 10:      # +
                result_val = v1 + v2
            elif op == 11:    # -
                result_val = v1 - v2
            elif op == 12:    # *
                result_val = v1 * v2
            elif op == 13:    # /
                result_val = v1 / v2
            elif op == 14:    # div
                result_val = v1 // v2
            elif op == 15:    # mod
                result_val = v1 % v2
            elif op == 16:    # exp
                result_val = v1 ** v2
            elif op == 17:    # lt
                result_val = (v1 < v2)
            elif op == 18:    # le
                result_val = (v1 <= v2)
            elif op == 19:    # gt
                result_val = (v1 > v2)
            elif op == 20:    # ge
                result_val = (v1 >= v2)
            elif op == 21:    # eq
                result_val = (v1 == v2)
            elif op == 22:    # ne
                result_val = (v1 != v2)
            else:
                result_val = 0

            set_value(res, result_val)
            IP += 1

        elif op == 2:  # mov
            val = get_value(a1)
            set_value(res, val)
            IP += 1

        elif op == 9:  # label – não faz nada em tempo de execução
            IP += 1

        elif op == 3:  # goto
            IP = res  # res é o índice da instrução alvo

        elif op == 4:  # if_false
            cond_val = get_value(a1)
            if not cond_val:
                IP = res  # res é o índice da instrução alvo
            else:
                IP += 1

        elif op == 5:  # param
            val = get_value(a1)
            param_buffer.append(val)
            IP += 1

        elif op == 6:  # call
            func_addr = a1   # índice da instrução de entrada da função
            n_args = a2      # número de argumentos

            # busca função pelo endereço
            target_func = None
            for fname, finfo in functions.items():
                if isinstance(finfo, dict) and finfo.get("addr") == func_addr:
                    target_func = finfo
                    break

            if target_func is None or func_addr < 0:
                raise RuntimeError(f"Função não declarada: {a1}")

            if n_args != target_func['param_count']:
                raise RuntimeError(
                    f"Chamada de função com {n_args} args, "
                    f"mas {target_func['param_count']} esperados."
                )

            # pega os argumentos do buffer
            args = param_buffer[-n_args:] if n_args > 0 else []
            if n_args > 0:
                del param_buffer[-n_args:]

            # empilha retorno e base antiga
            ret_ip = IP + 1
            old_BP = BP
            ret_dest = res

            stack_mem.append(ret_ip); SP += 1
            stack_mem.append(old_BP); SP += 1
            stack_mem.append(ret_dest); SP += 1

            # novo frame
            BP = SP

            # insere parâmetros no frame
            for val in args:
                stack_mem.append(val); SP += 1

            # aloca espaço para locais (frame_size = params + locais)
            frame_total = target_func['frame_size']
            local_count = frame_total - n_args
            for _ in range(local_count):
                stack_mem.append(0); SP += 1

            # desvia para a função
            IP = func_addr

        elif op == 7:  # ret
            ret_val = get_value(a1)

            # recupera info do frame atual (valores empilhados)
            ret_ip = stack_mem[BP - 3]
            prev_BP = stack_mem[BP - 2]
            ret_dest = stack_mem[BP - 1]

            # libera frame atual da pilha
            SP = BP - 3
            stack_mem = stack_mem[:SP]

            # escreve valor de retorno no frame chamador
            if ret_dest is not None:
                current_BP = BP
                BP = prev_BP
                set_value(ret_dest, ret_val)
                BP = current_BP

            # restaura base pointer e IP
            BP = prev_BP
            IP = ret_ip

            # se IP sair do código depois do retorno do main, termina
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


#  FUNÇÃO DE PARSE (USA O PLY)


def parse_code(src, dump_productions=False):
    """
    Faz o parse do código-fonte em Lisp usando PLY e retorna a AST.
    Também imprime a AST para debug.
    """
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


#  PIPELINE COMPLETO: PARSE -> CÓDIGO -> EXECUÇÃO


def executar_codigo(codigo_fonte: str, nome_arquivo_intermediario: str = "codigo_intermediario.txt"):
    """
    Executa todo o pipeline para um código em Lisp:
      1) faz o parse (gera AST)
      2) gera código intermediário (quádruplas)
      3) salva em arquivo (nome configurável)
      4) executa no interpretador
    """
    # 1) Parsing -> AST
    ast = parse_code(codigo_fonte, dump_productions=False)

    # 2) Geração de código intermediário
    cg = CodeGenerator()
    start_main = cg.gen_program(ast)

    print("\nCódigo intermediário (quádruplas <op, arg1, arg2, result>):\n")
    for i, quad in enumerate(cg.code):
        op, a1, a2, res = quad
        op_name = cg.rev_opcodes.get(op, op)
        print(f"{i:03}: ({op_name}, {a1}, {a2}, {res})")

    # grava no arquivo pedido (programa.txt -> codigo_intermediario.txt,
    # expressões interativas -> codigo_intermediario2.txt)
    cg.dump_to_file(nome_arquivo_intermediario)

    # 3) Executa no ambiente de execução
    print("\nSaída do programa (resultados das expressões):\n")
    run_intermediate(
        cg.code,
        cg.functions,
        start_main,
        data=cg.data,
        main_frame_size=cg.main_frame_size
    )
    print()  # linha em branco no final


#  MÓDULO PRINCIPAL


if __name__ == "__main__":
    # 1) Lê o programa principal de um arquivo
    try:
        with open("programa.txt", "r", encoding="utf-8") as f:
            source_code = f.read()
    except FileNotFoundError:
        print("Erro: arquivo 'programa.txt' não encontrado.")
        sys.exit(1)

    print("=== Executando programa do arquivo programa.txt ===\n")
    # aqui usa o arquivo padrão: codigo_intermediario.txt
    executar_codigo(source_code)

    # 2) Modo interativo: usuário digita expressões
    while True:
        print("\n")
        user_code = input("\n> ")

        if not user_code.strip():
            print("")
            break

        try:
            print("\n=== Executando expressão digitada ===\n")
            # aqui grava em codigo_intermediario2.txt
            executar_codigo(user_code, nome_arquivo_intermediario="codigo_intermediario2.txt")
        except Exception as e:
            print(f"Erro ao processar a expressão: {e}")
