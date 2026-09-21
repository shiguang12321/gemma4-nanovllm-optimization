from collections import deque#双端队列
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus#序列
from nanovllm.engine.block_manager import BlockManager#块管理器


class Scheduler:#调度器

    def __init__(self, config: "Config", tokenizer=None):
        self.max_num_seqs = config.max_num_seqs        #一步最多执行16条请求
        self.max_num_batched_tokens = config.max_num_batched_tokens   #一步最多执行16384个token
        self.eos = config.eos                               #结束符
        self.tokenizer = tokenizer                            #tokenizer
        self.block_size = config.kvcache_block_size            #块大小256个token
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)#块管理器
        self.waiting: deque[Sequence] = deque()                #等待队列
        self.running: deque[Sequence] = deque()                #运行队列

    def is_finished(self): #判断是否完成
        return not self.waiting and not self.running

    def add(self, seq: Sequence):#新请求排到waiting队列
        self.waiting.append(seq)

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = [] #已调度序列
        num_batched_tokens = 0 #已调度token数

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs: #等待队列不为空且已调度序列数小于最大序列数
            seq = self.waiting[0] #获取等待队列的第一个序列
            remaining = self.max_num_batched_tokens - num_batched_tokens #剩余token数
            if remaining == 0: #剩余token数为0，则跳出循环
                break
            if not seq.block_table: #如果序列的块表为空，则分配块
                num_cached_blocks = self.block_manager.can_allocate(seq) #分配块
                if num_cached_blocks == -1: #分配块失败，则跳出循环
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size #计算序列的token数
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens #计算序列的token数
            if remaining < num_tokens and scheduled_seqs:   #如果剩余token数小于序列的token数且已调度序列数大于0，则跳出循环
                break
            if not seq.block_table: #如果序列的块表为空，则分配块
                self.block_manager.allocate(seq, num_cached_blocks) #分配块
            seq.num_scheduled_tokens = min(num_tokens, remaining) #计算已调度token数
            num_batched_tokens += seq.num_scheduled_tokens #已调度token数累加
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens: #如果已调度token数等于序列的token数，则将序列状态设置为运行
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft() #将序列从等待队列中移除
                self.running.append(seq) #将序列添加到运行队列
            scheduled_seqs.append(seq) #将序列添加到已调度序列

        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = self._schedule_prefill()
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:#运行队列不为空且已调度序列数小于最大序列数
            seq = self.running.popleft() #获取运行队列的第一个序列
            while not self.block_manager.can_append(seq): #如果序列不能添加到块管理器，则抢占序列
                if self.running: #如果运行队列不为空，则抢占序列
                    self.preempt(self.running.pop()) #抢占序列
                else:
                    self.preempt(seq)
                    break
            else: #如果序列可以添加到块管理器，则添加到已调度序列
                seq.num_scheduled_tokens = 1 #已调度token数为1
                seq.is_prefill = False #预填充标志为False
                self.block_manager.may_append(seq) #添加到块管理器
                scheduled_seqs.append(seq) #将序列添加到已调度序列
        if not scheduled_seqs:
            # A preempted request can now prefill using the released blocks.
            # Retry once, not recursively: an oversized request must fail
            # explicitly rather than spin forever or disappear from queues.
            scheduled_seqs = self._schedule_prefill()
            if scheduled_seqs:
                return scheduled_seqs, True
            raise RuntimeError("No request can make progress: insufficient KV cache blocks")
        self.running.extendleft(reversed(scheduled_seqs)) #将已调度序列添加到运行队列
        return scheduled_seqs, False #返回已调度序列和False

    def preempt(self, seq: Sequence): #显存不够时怎么让座
        seq.status = SequenceStatus.WAITING #将序列状态设置为等待
        seq.is_prefill = True #预填充标志为True
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq) #释放块
        self.waiting.appendleft(seq) #将序列添加到等待队列

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool): #后处理序列
        for seq, token_id in zip(seqs, token_ids):#将序列和token id配对
            self.block_manager.hash_blocks(seq) #哈希块
            seq.num_cached_tokens += seq.num_scheduled_tokens #已缓存token数累加
            seq.num_scheduled_tokens = 0 #已调度token数为0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens: #如果预填充且已缓存token数小于序列的token数，则跳过
                continue
            seq.append_token(token_id) #添加token
            hit_stop = token_id in seq.stop_token_ids #判断是否命中停止符
            if not hit_stop and self.tokenizer is not None and seq.stop: #如果未命中停止符且tokenizer不为空且序列有停止符，则判断是否命中停止符
                text = self.tokenizer.decode(seq.completion_token_ids)#解码token id
                hit_stop = any(text.endswith(stop) for stop in seq.stop)#判断是否命中停止符 是否以停止符结尾
            if not hit_stop and self.tokenizer is not None and seq.stop_regex: #如果未命中停止符且tokenizer不为空且序列有停止符正则表达式，则判断是否命中停止符
                text = self.tokenizer.decode(seq.completion_token_ids)#解码token id
                hit_stop = re.search(seq.stop_regex, text) is not None#判断是否命中停止符 是否符合停止符正则表达式
            if (not seq.ignore_eos and token_id == self.eos) or hit_stop or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED #将序列状态设置为完成
                self.block_manager.deallocate(seq) #释放块
                self.running.remove(seq) #将序列从运行队列中移除
