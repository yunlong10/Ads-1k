from eval import Eval

def main():
    infer = Eval()
    your_results = ... # replace by your results
    given_times = [10,15]

    print('--------------------------------------------------------------------------------------------')
    print("|%12s|%14s|%10s|%10s|%12s|%14s|" % (
        " Given Time ", " #Test Sample ", " Imp@Time ",
        " Coh@Time ", " Time Score ", " Imp-Coh@Time "))
    print('--------------------------------------------------------------------------------------------')
    for i in given_times:
        lo, hi = int(i * 0.8), int(i * 1.2)
        infer.infer(results=your_results, given_time=(lo, hi))

if __name__ == '__main__':
    main()