from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:#一页物理内存

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:#块管理器 管理块的分配和释放

    def __init__(self, num_blocks: int, block_size: int): #num_blocks: 块数量 block_size: 块大小
        self.block_size = block_size  #一块能装多少token
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] #创建num_blocks个块，这是固定大小的块池，运行时不再扩容，只在池里分配回收
        self.hash_to_block_id: dict[int, int] = dict() #哈希到块id的映射  hash_blocks() 写入，can_allocate() / allocate() 查询。
        self.free_block_ids: deque[int] = deque(range(num_blocks)) #空闲块id列表
        self.used_block_ids: set[int] = set() #已使用块id集合
        self.cache_queries = 0 #缓存查询次数
        self.cache_hits = 0 #缓存命中次数
        self.cached_blocks = 0 #缓存块数量

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): #链式哈希  第 i 块的哈希 = hash(第 i-1 块的哈希 + 第 i 块的 token)
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())  #np.array(token_ids).tobytes() 把这 256 个 token id 变成原始字节，接着哈希。
        return h.intdigest()#只给满块打哈希，提前预留出一些空闲块，避免频繁分配回收

    def _allocate_block(self) -> int:#从空闲块id列表中取出一个空闲块id
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0  #如果不成立就直接报错停下来
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:#如果块的哈希不为-1且哈希到块id的映射不为空，则删除哈希到块id的映射
            del self.hash_to_block_id[block.hash] #删除哈希到块id的映射
        block.reset() #重置块
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):#将块id添加到空闲块id列表中
        assert self.blocks[block_id].ref_count == 0 #还块时，块引用数必须为0
        self.used_block_ids.remove(block_id) #从正在用的集合里删除这个编号
        self.free_block_ids.append(block_id) #添加到空闲块id列表中

    def can_allocate(self, seq: Sequence) -> int:#检查是否可以分配块 回答能复用几块，空闲块够不够
        self.cache_queries += 1 #缓存查询次数加1
        h = -1 #前一个块的哈希
        num_cached_blocks = 0 #能复用几块
        num_new_blocks = seq.num_blocks #新分配几块
        for i in range(seq.num_blocks - 1): #循环看前面的完整块
            token_ids = seq.block(i) #获取第i块的token id
            h = self.compute_hash(token_ids, h) #用“上一块的 hash + 本块 token”算出本块指纹，写回 h
            block_id = self.hash_to_block_id.get(h, -1) #用指纹查哈希到块id的映射，-1 表示“查不到”
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: #如果查不到或本块 token 和哈希到块id的映射不一致，则跳出循环
                break
            num_cached_blocks += 1 #能复用几块加1
            if block_id in self.used_block_ids: #如果块id在已使用块id集合里，则新分配几块减1
                num_new_blocks -= 1
        if num_cached_blocks:#如果能复用几块大于0，则缓存命中次数加1，缓存块数量加几块
            self.cache_hits += 1
            self.cached_blocks += num_cached_blocks
        if len(self.free_block_ids) < num_new_blocks: #如果空闲块不够，则返回-1
            return -1
        return num_cached_blocks #返回能复用几块

    def stats(self): #统计信息
        return dict(
            cache_queries=self.cache_queries,  #一共查过多少次缓存
            cache_hits=self.cache_hits,   #有多少次至少命中了一块前缀
            cached_blocks=self.cached_blocks, #复用了多少块
            free_blocks=len(self.free_block_ids), #空闲块数量
            used_blocks=len(self.used_block_ids), #已使用块数量
        )

    def allocate(self, seq: Sequence, num_cached_blocks: int): #把块挂到这条序列上
        assert not seq.block_table #如果序列的块表不为空，则抛出异常
        h = -1 #链式hash的起点
        for i in range(num_cached_blocks):  #只处理那些能复用的块
            token_ids = seq.block(i)   #取出第i块的token
            h = self.compute_hash(token_ids, h) #算出这一块的指纹
            block_id = self.hash_to_block_id[h]  #取出物理块号
            block = self.blocks[block_id]  #取出物理块
            if block_id in self.used_block_ids: #如果已经在用，不用新开，两条共用一条KV
                block.ref_count += 1
            else:   #如果不在用，则新开，一条独占一条KV
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):   #复用的块已经挂好了，给每命中的块开新页
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence): #这条序列不用这些块了
        for block_id in reversed(seq.block_table):   #从后往前释放，和分配顺序相反
            block = self.blocks[block_id]      #取出物理块对象
            block.ref_count -= 1  #引用计数减1
            if block.ref_count == 0:   #如果引用计数为0，则释放物理块
                self._deallocate_block(block_id)  #释放物理块
        seq.num_cached_tokens = 0  #计数清零
        seq.block_table.clear()  #块表清空

    def can_append(self, seq: Sequence) -> bool: #下次decode要不要新开一块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)  #

    def may_append(self, seq: Sequence): #需要的话开一块挂上
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence): #给满块打哈希
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return  #如果起始块和结束块相同，则返回
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end): #循环给满块打哈希
            block = self.blocks[seq.block_table[i]] #取出物理块对象
            token_ids = seq.block(i) #取出第i块的token id
            h = self.compute_hash(token_ids, h) #算出这一块的指纹
            block.update(h, token_ids) #更新物理块
            self.hash_to_block_id[h] = block.block_id #更新哈希到块id的映射
