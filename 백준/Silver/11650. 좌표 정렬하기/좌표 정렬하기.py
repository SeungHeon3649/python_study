num = int(input())
x = []
for i in range(num):
    corr = list(map(int, input().split()))
    x.append(corr)
x = sorted(x, key = lambda x : (x[0], x[1]))
for i, j in x:
    print(i, j)