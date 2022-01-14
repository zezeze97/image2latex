
def minRemoveToMakeValid(latex,left_symbel,right_symbel):
    temp_stack = []
    i = 0
    N = len(latex)
    while i < N:
        if latex[i] == right_symbel:
            if len(temp_stack) != 0:
                latex = latex[:i] + latex[i+1:]
                N -= 1
                i -= 1
            else:
                temp_stack.pop()
        elif latex[i] == left_symbel:
            temp_stack.append(i)
    while len(temp_stack) != 0:
        i = temp_stack.pop()
        latex = latex[:i] + latex[i+1:]
    return latex


if __name__ == '__main__':
    raw_latex = '/alpha_{1'
    prcessed_latex = minRemoveToMakeValid(raw_latex,'{','}')
    print(prcessed_latex)