

def process_it(it):
    print ("test!!!!!!!!!!!!!")
    b = it 
    while True:
        try:
            tmp_it = next(b)
            a1 = tmp_it[0]
            a2 = tmp_it[1]
            print (a1)
            print (a2)
        except StopIteration:
            break

    print ("test copy")
    while True:
        try:
            tmp_it = next(b)
            a1 = tmp_it[0]
            a2 = tmp_it[1]
            print (a1)
            print (a2)
        except StopIteration:
            break

