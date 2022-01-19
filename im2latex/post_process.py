
def minRemoveToMakeValid(latex,left_symble,right_symble):
    left_num = 0
    right_num = 0
    for token in latex:
        if token == left_symble:
            left_num += 1
        if token == right_symble:
            right_num += 1
    if left_num > right_num:
        latex += right_symble*(left_num - right_num)
    if right_num > left_num:
        latex = left_symble*(right_num - left_num) + latex
    return latex
    


if __name__ == '__main__':
    raw_latex = r'X_}'
    prcessed_latex = minRemoveToMakeValid(raw_latex,'{','}')
    prcessed_latex = minRemoveToMakeValid(prcessed_latex,'(',')')
    print(prcessed_latex)