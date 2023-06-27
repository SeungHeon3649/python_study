def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)

num = int(input())
for i in range(num):
    a, b = map(int, input().split())
    result = 1
    if a == b:
        result = 1
    else:
        result = int(factorial(b) / (factorial(b - a) * factorial(a)))
    print(result)