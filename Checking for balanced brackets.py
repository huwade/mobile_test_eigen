

def match(expression):
    mapping = dict(zip('([{', ')]}'))
    queue   = []
    for cont in expression:
        if cont in mapping:
            queue.append(mapping[cont])
        elif not (cont == queue.pop()):
            return false
    return not queue




if __name__ == '__main__':
    expression = ('(()){[]')
    print(match(expression))
        
        
