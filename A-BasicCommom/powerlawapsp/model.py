# matrix sparseness judgment, convergence judgment, distance product by matrix multiplication, association law for distance product,
#   scale-free characteristics of the social networks degrees and the precision limit of floating point operations on the modern hardware.
# 1. DPMM: distance product by MM
# 2. Association law of distance product
# 3. Diameter Limit: scale-free network
# 4. Precision Limit: n^(epochs) < max_float for intermediate value:float32, float64
# 5. Convergence judgement.
# 6. Sparseness judgement
import math
import cupy
import cupyx
import numpy
import scipy
from cupy import cusparse
from scipy.sparse import csr_matrix


class Device:
    def __init__(self, config):
        self.use = config.device
        if 'threshold' in config.__dict__.keys():
            self.THRESHOLD = config.threshold

        if config.device == 'cpu':
            self.device = numpy
            self.csr_matrix = scipy.sparse.csr_matrix
        else:
            self.device = cupy
            self.csr_matrix = cupyx.scipy.sparse.csr_matrix


# 1. DPMM: distance product by MM
# 2. Association law of distance product
# 3. Diameter Limit: scale-free network
# 4. Precision Limit: n^(epochs) < max_float for intermediate value:float32, float64
# 5. Convergence judgement.
class APSP(Device):

    def __init__(self, matrix, config):
        super().__init__(config)
        if self.use == 'cpu':
            self.adj_matrix =matrix
        else:
            # load into gpu
            self.adj_matrix = cupy.array(matrix)
        self.e_max = self.device.max(self.adj_matrix)
        self.g_diameter = config.diameter
        self.use_dynamic = config.converge_check
        self.epsilon = config.lr
        print('shape:', self.adj_matrix.shape, 'element_max', self.e_max, 'diameter:', self.g_diameter, 'use_dynamic:', self.use_dynamic)

    def stat(self, op):
        stat = [len(op[self.device.where(op <= i)]) for i in (1, self.g_diameter, self.e_max)]
        print('stat:',(stat[0], stat[1]-stat[0], stat[2]-stat[1]))
        return stat

    def max(self, op):
        print('mmax')
        index_op = self.device.where(op < self.e_max)
        op_min = self.device.min(op[index_op])
        op_max = self.device.max(op[index_op])
        print('minv is ', op_min , 'maxv is ', op_max)
        return op_max

    def exponent(self, op, base, current_maxv):
        print('exp')
        index_op = self.device.where(op < self.e_max)
        rindex_op = self.device.where(op >= self.e_max)
        print('expi:',len(op[index_op]),'ri:',len(op[rindex_op]))
        op[index_op] = current_maxv - op[index_op]
        op[index_op] = self.device.power(base + 1, op[index_op])
        op[rindex_op] = 0
        print('exp:',op)
        return op

    def logarithm(self, op, base, current_maxv):
        print('log')
        index_zero = self.device.where(op>0)
        rindex_zero = self.device.where(op==0)
        print('logi:', len(op[index_zero]), 'ri:', len(op[rindex_zero]))
        op[index_zero] = 2 * current_maxv - self.device.log(op[index_zero]) // self.device.log(base + 1)
        op[rindex_zero] = self.e_max
        print('log',op)
        return op

    # 1. DPMM algorithm
    def dp(self, op):
        print('dp')
        m = op.shape[0]
        op_max = self.max(op)
        op = self.exponent(op, m, op_max)
        op = self.device.matmul(op,op)
        op = self.logarithm(op, m, op_max)
        print('dp:',op)
        return op

    # 1. DPMM
    # 2. Association law for DP.
    # 5. Convergence judgement.
    def apsp(self, g_diameter=9):
        print('apsp')
        adj = self.adj_matrix
        counter = math.ceil(math.log(g_diameter, 2))
        print('LOOP N:',counter)
        # 2. Association law
        for i in range(counter):
            print('loop index:', i)
            print('apsp,a:', adj)
            # 1. DPMM
            wr = self.dp(adj.copy())
            print('apsp,b:', wr)
            post = self.device.minimum(adj, wr)
            print('apsp,c:', adj)
            # 5. Convergence judgement
            if self.use_dynamic:
                print('checking diff:')
                equalsum = self.device.sum(self.device.equal(adj, post))
                print('equals:', equalsum, "/", self.device.size(adj)," ({}%)".format(equalsum*100.0/ self.device.size(adj)))
                if equalsum > (1.0 - self.epsilon) * self.device.size(adj):
                    print('LOOP EXIT by dynamic decision. at LOOP:', i)
                    break
            adj = post
        print('apsp:', adj)
        return adj

    def apsp_iter(self, g_diameter=9):
        print('apsp')
        adj = self.adj_matrix
        counter = math.ceil(math.log(g_diameter, 2))
        print('LOOP N:',counter)
        for i in range(counter):
            print('loop index:', i)
            print('apsp,a:', adj)
            wr = self.dp(adj.copy())
            print('apsp,b:', wr)
            post = self.device.minimum(adj, wr)
            print('apsp,c:', adj)
            if self.use_dynamic and self.device.all(self.device.equal(adj, post)):
                yield adj
                print('LOOP EXIT by dynamic decision.')
                break
            adj = post
            yield  post
        print('apsp:FIN')


# 1. DPMM: distance product by MM
# 2. Association law of distance product
# 3. Diameter Limit: scale-free network
# 4. Precision Limit: n^(epochs) < max_float for intermediate value:float32, float64
# 5. Convergence judgement.
# 6. Sparseness judgement
class APSPPowerLawBound(Device):

    def __init__(self,matrix, config):
        super().__init__(config)
        if self.use == 'cpu':
            self.adj_matrix = matrix
        else:
            # load into gpu
            self.adj_matrix = cupy.array(matrix)
        self.e_max = self.device.max(self.adj_matrix)
        self.g_diameter = config.diameter
        self.use_dynamic = config.converge_check
        self.use_sparse = config.sparse_check
        self.epsilon = config.lr
        print('shape:', self.adj_matrix.shape, 'element_max', self.e_max, 'diameter:', self.g_diameter, 'use_dynamic:', self.use_dynamic)

    def stat(self, op):
        stat = [len(op[self.device.where(op <= i)]) for i in (1, self.g_diameter, self.e_max)]
        print('stat:',(stat[0], stat[1]-stat[0], stat[2]-stat[1]))
        return stat

    def max(self, op):
        print('mmax')
        index_op = self.device.where(op < self.e_max)
        op_min = self.device.min(op[index_op])
        op_max = self.device.max(op[index_op])
        print('minv is ', op_min , 'maxv is ', op_max)
        return op_max

    def exponent(self, op, base, current_maxv):
        print('exp')
        index_op = self.device.where(op < self.e_max)
        rindex_op = self.device.where(op >= self.e_max)
        print('expi:',len(index_op[0]),'ri:',len(rindex_op[0]))
        op[index_op] = current_maxv - op[index_op]
        op[index_op] = self.device.power(base + 1, op[index_op])
        op[rindex_op] = 0
        print('exp:',op)
        if self.use_sparse:
            self.density = float(len(index_op[0]))/float(len(index_op[0])+len(rindex_op[0]))
        return op

    def logarithm(self, op, base, current_maxv):
        print('log')
        index_zero = self.device.where(op>0)
        rindex_zero = self.device.where(op==0)
        print('logi:', len(index_zero[0]), 'ri:', len(rindex_zero[0]))
        op[index_zero] = 2 * current_maxv - self.device.log(op[index_zero]) // self.device.log(base + 1)
        op[rindex_zero] = self.e_max
        print('log',op)
        return op

    # 1. DPMM
    # 6. Sparseness judgement
    def dp(self, op):
        print('dp')
        m = op.shape[0]
        op_max = self.max(op)
        op = self.exponent(op, m, op_max)
        if self.use_sparse:
            print('dense', self.density)
        # check op is dense or not, within THRESHOLD such as 10% sparse, then decide to use MM or SPMM.
        if self.use_sparse and self.density < self.THRESHOLD:
            sop = self.csr_matrix(op)
            print('sparse nnz:', sop.nnz)
            if self.use == 'cpu':#cpu use @ after python 3
                sop = sop @ sop
            else:#gpu use cusparse.csrgemm
                sop = cusparse.csrgemm(sop, sop)
            print('sparse nnz2:', sop.nnz)
            op = sop.todense()
        else:
            op = self.device.matmul(op, op)
        op = self.logarithm(op, m, op_max)
        print('dp:',op)
        return op

    # 1. DPMM
    # 2. Association law for DP.
    # 5. Convergence judgement.
    # 6. Sparseness judgement.
    def apsp(self, g_diameter=9):
        print('apsp')
        adj = self.adj_matrix
        counter = math.ceil(math.log(g_diameter, 2))
        print('LOOP N:',counter)
        for i in range(counter):
            print('loop index:', i)
            print('apsp,a:', adj)
            wr = self.dp(adj.copy())
            print('apsp,b:', wr)
            post = self.device.minimum(adj, wr)
            print('apsp,c:', adj)
            if self.use_dynamic:
                print('checking diff:')
                equalsum = self.device.sum(self.device.equal(adj, post))
                print('equals:', equalsum, "/", self.device.size(adj)," ({}%)".format(equalsum*100.0/ self.device.size(adj)))
                if equalsum > (1.0 - self.epsilon) * self.device.size(adj):
                    print('LOOP EXIT by dynamic decision. at LOOP:', i)
                    break
            adj = post
        print('apsp:', adj)
        return adj

    def apsp_iter(self, g_diameter=9):
        print('apsp')
        adj = self.adj_matrix
        counter = math.ceil(math.log(g_diameter, 2))
        print('LOOP N:',counter)
        for i in range(counter):
            print('loop index:', i)
            print('apsp,a:', adj)
            wr = self.dp(adj.copy())
            print('apsp,b:', wr)
            post = self.device.minimum(adj, wr)
            print('apsp,c:', adj)
            if self.use_dynamic and self.device.all(self.device.equal(adj, post)):
                yield adj
                print('LOOP EXIT by dynamic decision.')
                break
            adj = post
            yield  post
        print('apsp:FIN')

