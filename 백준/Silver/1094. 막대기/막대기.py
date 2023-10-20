BASE = 64
x = int(input())
cnt = 0
while True:
    if x == 0: break
    if x < BASE:
        BASE = BASE // 2
    else:
        x = x - BASE
        cnt += 1
print(cnt)