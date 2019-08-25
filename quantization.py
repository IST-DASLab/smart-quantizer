from bisect import bisect_left
import random
import math


class Quantizer:
    def __init__(self, k, bucket_size):
        self.k = k
        self.bucket_size = bucket_size
        self.variance = 0

    def quantize_bucket(self, a):
        raise NotImplementedError

    def quantize(self, a):
        if self.bucket_size == -1:
            return self.quantize_bucket(a)
        quantized = []
        # print("Length %d" % len(a))
        for i in range((len(a) + self.bucket_size - 1) // self.bucket_size):
            # print("quantize %d out of %d" % (i, (len(a) + self.bucket_size - 1) // self.bucket_size))
            quantized += self.quantize_bucket(a[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))])
        return quantized


class SmartQuantizer(Quantizer):
    def __init__(self, k, bucket_size):
        super(SmartQuantizer, self).__init__(k, bucket_size)

    def error(self, i, j):
        return -(j - i + 1) * self.a[i] * self.a[j] \
              + (self.a[i] + self.a[j]) * (self.sum[j + 1] - self.sum[i])\
              - (self.sumsq[j + 1] - self.sumsq[i])

    def compute(self, k, L, R, optL, optR):
        if L == R - 1:
            return
        M = (L + R) // 2
        opt = None
        for j in range(optL, min(optR, M) + 1):
            res = self.dp[k - 1][j] + self.error(j, M)
            if res < self.dp[k][M]:
                self.dp[k][M] = res
                self.prev[k][M] = j
                opt = j
        self.compute(k, L, M, optL, opt)
        self.compute(k, M, R, opt, optR)

    def compute2(self, k):
        for i in range(len(self.dp[k])):
            for j in range(i + 1):
                res = self.dp[k - 1][j] + self.error(j, i)
                if res < self.dp[k][i]:
                    self.dp[k][i] = res
                    self.prev[k][i] = j

    def sparsify(self, a):
        choose = int(max(0.01 * len(a), 512))
        if choose > len(a):
            return a
        if choose > len(a) / 2:
            taken = [True]*len(a)
            for i in range(len(a) - choose):
                x = random.randint(0, len(a) - 1)
                while not taken[x]:
                    x = random.randint(0, len(a) - 1)
                taken[x] = False
        else:
            taken = [False]*len(a)
            for i in range(choose):
                x = random.randint(0, len(a) - 1)
                while taken[x]:
                    x = random.randint(0, len(a) - 1)
                taken[x] = True

        sparsification = [0]*choose
        id = 0
        for i in range(len(a)):
            if taken[i]:
                sparsification[id] = a[i]
                id += 1
        return sparsification

    def quantization_points(self, a, k):
        k -= 1

        self.a = a.copy()
        self.a = self.sparsify(a)
        self.a.sort()
        self.a[0] = min(a)
        self.a[len(self.a) - 1] = max(a)
        self.sum = [0]*(len(self.a) + 1)
        self.sumsq = [0]*(len(self.a) + 1)
        for i in range(len(self.sum) - 1):
            self.sum[i + 1] = self.sum[i] + self.a[i]
            self.sumsq[i + 1] = self.sumsq[i] + self.a[i] * self.a[i]

        self.dp = [[float('inf')]*len(self.a) for _ in range(k + 1)]
        self.prev = [[-1]*len(self.a) for _ in range(k + 1)]
        self.dp[0][0] = 0
        for i in range(k):
            self.compute(i + 1, 0, len(self.dp[i]), 0, len(self.a))
        # ans = self.dp[k][len(self.a) - 1]

        # self.dp = [[float('inf')]*len(self.a) for _ in range(k + 1)]
        # self.prev = [[-1]*len(self.a) for _ in range(k + 1)]
        # self.dp[0][0] = 0
        # for i in range(k):
        #     self.compute2(i + 1)
        # if ans != self.dp[k][len(self.a) - 1]:
        #     print("Fail!")

        p = len(self.a) - 1
        points = [0]*(k + 1)
        points[k] = self.a[p]
        for i in range(k, 0, -1):
            p = self.prev[i][p]
            points[i - 1] = self.a[p]

        return points

    def quantize_bucket(self, a):
        min_a = min(a)
        max_a = max(a)
        points = self.quantization_points(a, self.k)

        # variance1 = 0
        # for i in range(len(a)):
        #     pos = bisect_left(points, a[i])
        #     if pos == 0:
        #         pos += 1
        #     variance1 += (a[i] - points[pos - 1]) * (points[pos] - a[i])
        #
        # variance2 = 0
        # fmin = min(a)
        # fmax = max(a)
        # for i in range(len(a)):
        #     if fmax == fmin:
        #         continue
        #     unit = (fmax - fmin) / (self.k - 1)
        #     v = math.floor((a[i] - fmin) / unit)
        #     variance2 += (a[i] - (fmin + v * unit)) * ((fmin + (v + 1) * unit) - a[i])
        # if variance1 > variance2:
        #     print("Fail!")

        # print(points)
        res = [0] * len(a)
        for i in range(len(a)):
            pos = bisect_left(points, a[i])
            if pos == 0:
                pos += 1
            if pos == len(points):
                pos -= 1
            fraction = random.random()
            self.variance += (a[i] - points[pos - 1]) * (points[pos] - a[i])
            if (a[i] - points[pos - 1]) < fraction * (points[pos] - points[pos - 1]):
                res[i] = points[pos]
            else:
                res[i] = points[pos - 1]
        return res


class StandardQuantizer(Quantizer):
    def __init__(self, k, bucket_size):
        super(StandardQuantizer, self).__init__(k, bucket_size)
        self.k = k
        self.bucket_size = bucket_size

    def quantize_bucket(self, a):
        fmin = min(a)
        fmax = max(a)
        res = [0] * len(a)
        for i in range(len(a)):
            unit = (fmax - fmin) / (self.k - 1)
            if fmax - fmin == 0:
                q = fmin
            else:
                v = math.floor((a[i] - fmin) / unit + random.random())
                q = fmin + v * unit
                if q > a[i]:
                    l = q - unit
                    r = q
                else:
                    l = q
                    r = q + unit
                self.variance += (a[i] - l) * (r - a[i])
            res[i] = q
        return res


class SanityQuantizer(Quantizer):
    def __init__(self, k, bucket_size):
        super.__init__(k, bucket_size)
        self.k = k
        self.bucket_size = bucket_size

    def quantize_bucket(self, a):
        return [0] * len(a)


class HadamardQuantizer(Quantizer):
    def __init__(self, k, bucket_size):
        super(HadamardQuantizer, self).__init__(k, bucket_size)
        self.k = k
        self.bucket_size = bucket_size

    def hadamard_transform(self, a):
        h = 1
        while h < len(a):
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
                    a[j] /= math.sqrt(2)
                    a[j + h] /= math.sqrt(2)
            h *= 2

    def pad(self, a):
        k = 1
        pow = 1
        while k < len(a):
            k *= 2
            pow += 1
        res = [0]*k
        for i in range(len(a)):
            res[i] = a[i]
        return res

    def sign(self, x):
        return -1 if x < 0 else 1

    def quantize_bucket(self, a):
        b = self.pad(a)
        D = [self.sign(random.random() - 0.5) for _ in b]
        for i in range(len(b)):
            b[i] *= D[i]
        self.hadamard_transform(b)

        fmin = min(b)
        fmax = max(b)
        for i in range(len(b)):
            unit = (fmax - fmin) / (self.k - 1)
            if fmax - fmin == 0:
                q = fmin
            else:
                v = math.floor((b[i] - fmin) / unit + random.random())
                q = fmin + v * unit
            b[i] = q

        self.hadamard_transform(b)

        for i in range(len(b)):
            b[i] *= D[i]

        b = b[0:len(a)]

        # sanity check
        # mse_hadamard = 0
        # for i in range(len(b)):
        #     mse_hadamard += (a[i] - b[i])**2
        #
        # mse_standard = 0
        # c = StandardQuantizer(self.k, self.bucket_size).quantize_bucket(a)
        # for i in range(len(a)):
        #     mse_standard += (a[i] - c[i])**2
        #
        # print(mse_hadamard, mse_standard)
        return b


class ExponentialQuantizer(Quantizer):
    def __init__(self, k, bucket_size):
        super(ExponentialQuantizer, self).__init__(k, bucket_size)
        self.k /= 2

    def quantize_bucket(self, a):
        vnorm = 0
        for x in a:
            vnorm += x * x
        if (vnorm == 0):
            vnorm = 1e-11
        else:
            vnorm = math.sqrt(vnorm)

        sign = [(1 if x >= 0 else -1) for x in a]

        a = [math.fabs(x) / vnorm for x in a]
        logs = [math.log2(x) if x != 0 else -32 for x in a]

        max_pow = int(max(logs))
        min_pow = max_pow - self.k + 2

        for i in range(len(a)):
            now = min(max_pow, max(min_pow, logs[i]))
            l = math.pow(2, int(now) - 1)
            r = math.pow(2, int(now))

            if a[i] < l:
                self.variance += (a[i] - 0) * (l - a[i]) * vnorm * vnorm
            elif a[i] > r:
                self.variance += (a[i] - r) * (1 - a[i]) * vnorm * vnorm
            else:
                self.variance += (max(a[i], l) - l) * (min(a[i], r) - a[i]) * vnorm * vnorm

            a[i] = min(r, max(a[i], l))
            if (a[i] - l) / (r - l) < random.random():
                a[i] = l
            else:
                a[i] = r
            a[i] = sign[i] * a[i] * vnorm
        return a
