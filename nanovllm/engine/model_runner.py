import torch
import torch.distributed as dist
from time import perf_counter
import warnings

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.gemma4 import Gemma4ForCausalLM, MOE_TRITON_DECODE_MAX_TOKENS
from nanovllm.engine.decode_plan import (
    common_cached_prefix_tokens, graph_batch_limit,
    parse_graph_block_buckets, select_graph_key,
)
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int = 0): #把一张 GPU 从「空卡」变成「可以跑推理」的状态。
        self.config = config #把配置挂到实例上
        hf_config = config.hf_config    #取出解析的模型结构配置。
        self.block_size = config.kvcache_block_size  #取出KV缓存块大小，
        self.enforce_eager = config.enforce_eager  #是否禁用CUDA Graph
        self.world_size = config.tensor_parallel_size
        self.rank = rank  #进程id
        self.graph_max_bs = graph_batch_limit(
            config.max_num_seqs, config.moe_impl, MOE_TRITON_DECODE_MAX_TOKENS)
        if not self.enforce_eager and config.max_num_seqs > self.graph_max_bs:
            warnings.warn(
                f"moe_impl='auto': CUDA Graph supports at most {self.graph_max_bs} tokens; "
                "larger decode batches use eager execution. Set moe_impl='grouped' "
                "to enable the optional capturable grouped path.", RuntimeWarning)

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank) #初始化PyTorch的分布式进程组
        torch.cuda.set_device(rank) #设置当前进程使用的GPU
        default_dtype = torch.get_default_dtype() #获取默认的浮点数类型
        torch.set_default_dtype(hf_config.dtype) #设置默认的浮点数类型
        torch.set_default_device("cuda") #设置默认的设备为GPU
        self.model = self.get_model(hf_config) #获取模型
        load_model(self.model, config.model) #加载模型
        self.sampler = Sampler() #创建采样器
        self.warmup_model() #预热模型
        self.allocate_kv_cache() #分配KV缓存
        if not self.enforce_eager: #启用 CUDA Graph 时预热并捕获
            self.capture_cudagraph()
        torch.set_default_device("cpu") #设置默认的设备为CPU
        torch.set_default_dtype(default_dtype) #设置默认的浮点数类型

    def get_model(self, hf_config): #只使用gemma4文本模型
        assert hf_config.model_type == "gemma4_text"
        return Gemma4ForCausalLM(hf_config)

    def exit(self):#释放GPU和分布式进程组
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def warmup_model(self): #用一次「最大规模假 prefill」预热并测峰值显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats() #重置峰值显存统计
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len #取出最大批量token数和最大模型长度
        seq_len = min(max_num_batched_tokens, max_model_len) #取最小值
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs) #取最小值
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)] #创建序列
        for seq in seqs: #设置序列的token数量
            seq.num_scheduled_tokens = seq_len #设置序列的token数量
        self.run(seqs, True) #运行模型
        torch.cuda.empty_cache() #清空缓存

    def allocate_kv_cache(self): #分配KV缓存
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info() #获取当前GPU的空闲内存和总内存
        used = total - free #计算已使用内存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] #获取峰值显存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] #获取当前显存
        attn_modules = [m for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")] #找到所有带kv的attention模块
        block_bytes = sum(2 * self.block_size * m.num_kv_heads * m.head_dim * hf_config.dtype.itemsize for m in attn_modules) #一个block占用多少字节
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes #计算需要分配的块数量
        assert config.num_kvcache_blocks > 0 #如果需要分配的块数量小于0，则抛出异常
        self.kv_cache = [] #创建KV缓存列表
        for module in attn_modules: #创建KV缓存
            shape = (config.num_kvcache_blocks, self.block_size, module.num_kv_heads, module.head_dim) #创建KV缓存形状
            module.k_cache = torch.empty(shape, dtype=hf_config.dtype) #创建KV缓存
            module.v_cache = torch.empty(shape, dtype=hf_config.dtype) #创建KV缓存
            self.kv_cache.extend([module.k_cache, module.v_cache]) #添加到KV缓存列表

    def prepare_block_tables(self, seqs: list[Sequence]): #把变长页表垫成矩阵
        max_len = max(len(seq.block_table) for seq in seqs) #找到最长的页表
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs] #把变长页表垫成矩阵
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True) #转换为张量并拷贝到GPU
        return block_tables #返回块表

    def prepare_prefill(self, seqs: list[Sequence]): #拼变长 prefill batch
        input_ids = [] #token id列表
        positions = [] #position列表
        cu_seqlens_q = [0] #query的cu_seqlens列表
        cu_seqlens_k = [0] #key的cu_seqlens列表
        max_seqlen_q = 0 #query的最大长度
        max_seqlen_k = 0 #key的最大长度
        slot_mapping = [] #slot mapping列表
        block_tables = None #block tables列表

        for seq in seqs:  #遍历序列
            start = seq.num_cached_tokens #起始位置
            seqlen_q = seq.num_scheduled_tokens #本步要跑的token数
            end = start + seqlen_q #跑完后这条序列的长度
            seqlen_k = end #key长度
            input_ids.extend(seq[start:end]) #添加token id
            positions.extend(range(start, end)) #添加position
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) #添加query的cu_seqlens
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) #添加key的cu_seqlens
            max_seqlen_q = max(seqlen_q, max_seqlen_q) #更新query的最大长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k) #更新key的最大长度
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size  #按页生成slot_mapping
            end_block = (end + self.block_size - 1) // self.block_size

        #slot_mapping：写路径。 每个新 token 一个整数，store_kvcache 做 cache[slot] = k[j]。 block_tables：读路径。 每个 token 一个整数，load_kvcache 做 k[j] = cache[slot]。

            for i in range(start_block, end_block):#遍历页表，生成slot_mapping
                slot_start = seq.block_table[i] * self.block_size #页起始slot
                if i == start_block: #第一页
                    slot_start += start % self.block_size #页内偏移
                if i != end_block - 1: #最后一页
                    slot_end = seq.block_table[i] * self.block_size + self.block_size #页结束slot
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size #页结束slot
                slot_mapping.extend(range(slot_start, slot_end)) #添加slot_mapping


        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 要不要带页表
            block_tables = self.prepare_block_tables(seqs)
        cu_seqlens_q_cpu = tuple(cu_seqlens_q)
        cu_seqlens_k_cpu = tuple(cu_seqlens_k)
        block_tables_cpu = tuple(tuple(seq.block_table) for seq in seqs) if block_tables is not None else None
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cpu", pin_memory=True).cuda(non_blocking=True) #搬到GPU并写入context
        positions = torch.tensor(positions, dtype=torch.int64, device="cpu", pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    cu_seqlens_q_cpu=cu_seqlens_q_cpu, cu_seqlens_k_cpu=cu_seqlens_k_cpu,
                    block_tables_cpu=block_tables_cpu)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]): #准备解码 每条序列送 1 个 token 进模型，告诉 attention 历史有多长、新 KV 写到哪个槽

    #prepare_decode 运行时，序列里已经包含「本步要算的那个 token」，但还不包含即将采样出来的下一个 token。
        cascade_prefix_len = None
        if self.config.enable_cascade_decode:
            prefix_tokens = common_cached_prefix_tokens(
                [seq.block_table for seq in seqs],
                [min(seq.num_cached_tokens, len(seq) - 1) for seq in seqs], self.block_size)
            # Fixed-shape metadata is consumed by the prefix AND suffix kernels.
            cascade_prefix_len = torch.tensor([prefix_tokens], dtype=torch.int32,
                                             device="cpu", pin_memory=True).cuda(non_blocking=True)
        input_ids = [] #last_token
        positions = [] #这个 token 的绝对位置（给 RoPE）
        slot_mapping = [] #新 KV 写到哪个槽
        context_lens = [] #历史有多长
        for seq in seqs:
            input_ids.append(seq.last_token) #添加last_token
            positions.append(len(seq) - 1) #添加绝对位置
            context_lens.append(len(seq)) #添加历史长度
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1) #添加新 KV 写到哪个槽

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cpu", pin_memory=True).cuda(non_blocking=True) #搬到GPU并写入context
        positions = torch.tensor(positions, dtype=torch.int64, device="cpu", pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables, cascade_prefix_len=cascade_prefix_len)
        return input_ids, positions
#用 input_ids 算出这 1 个 token 的 Q/K/V  store_kvcache：按 slot_mapping[i] 把该 token 的 K/V 写入 cache  按 block_tables[i] + context_lens[i] 从 cache 读出长度为 L 的历史（含刚写入的当前 token）做注意力



    def prepare_sample(self, seqs: list[Sequence]): #要不要准备温度向量
        if all(seq.temperature <= 1e-10 for seq in seqs):
            return None
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device="cpu", pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        context = get_context()
        key = None
        if not is_prefill and not self.enforce_eager:
            key = select_graph_key(input_ids.size(0), context.block_tables.size(1),
                                   self.graph_bs, self.graph_block_buckets)
        graph = self.graphs.get(key) if key is not None else None
        if graph is None:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)
        actual_blocks = context.block_tables.size(1)
        graph_vars = self.graph_vars
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][:bs, :actual_blocks] = context.block_tables
        if self.config.enable_cascade_decode:
            graph_vars["cascade_prefix_len"].copy_(context.cascade_prefix_len)
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]: #模型运行完整流水线
        try:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill)
            return self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = self.graph_max_bs
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.graph_bs = parse_graph_block_buckets(max_bs)
        self.graph_block_buckets = parse_graph_block_buckets(max_num_blocks, config.graph_block_buckets)
        num_graphs = len(self.graph_bs) * len(self.graph_block_buckets)
        if num_graphs > 64:
            warnings.warn(f"Capturing {num_graphs} CUDA Graphs may use substantial memory and startup time; "
                          "consider fewer graph_block_buckets.", RuntimeWarning)
        print(f"CUDA Graph: capturing {num_graphs} graphs "
              f"(batch={self.graph_bs}, blocks={self.graph_block_buckets})", flush=True)
        started = perf_counter()
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        cascade_prefix_len = (torch.zeros(1, dtype=torch.int32)
                              if config.enable_cascade_decode else None)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graphs = {}
        self.graph_pool = None
        # Each compiled elementwise shape and Triton variant is first invoked
        # OUTSIDE capture. Use one side stream, as required by CUDA Graph warmup.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        try:
            for bs in reversed(self.graph_bs):
                for num_blocks in reversed(self.graph_block_buckets):
                    graph = torch.cuda.CUDAGraph()
                    set_context(False, slot_mapping=slot_mapping[:bs],
                                context_lens=context_lens[:bs],
                                block_tables=block_tables[:bs, :num_blocks],
                                cascade_prefix_len=cascade_prefix_len)
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(2):
                            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                    torch.cuda.current_stream().wait_stream(warmup_stream)
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph, pool=self.graph_pool, stream=warmup_stream):
                        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                    if self.graph_pool is None:
                        self.graph_pool = graph.pool()
                    self.graphs[(bs, num_blocks)] = graph
                    torch.cuda.synchronize()
        finally:
            reset_context()
        self.graph_vars = dict(input_ids=input_ids, positions=positions,
                               slot_mapping=slot_mapping, context_lens=context_lens,
                               block_tables=block_tables, outputs=outputs)
        if cascade_prefix_len is not None:
            self.graph_vars["cascade_prefix_len"] = cascade_prefix_len
        self.graph_capture_seconds = perf_counter() - started
        print(f"CUDA Graph: captured {len(self.graphs)} graphs in "
              f"{self.graph_capture_seconds:.2f}s (includes warmup)", flush=True)
