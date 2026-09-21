import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):#初始化LLM引擎
        config_fields = {field.name for field in fields(Config)}#获取Config类中的所有字段
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}#获取kwargs中在Config类中的字段
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.model_runner = ModelRunner(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id#设置EOS token id
        self.scheduler = Scheduler(config, self.tokenizer)#初始化调度器
        atexit.register(self.exit)#注册退出函数

    def exit(self):
        if not hasattr(self, "model_runner"):#如果模型运行器不存在，则返回
            return#如果模型运行器不存在，则返回
        self.model_runner.exit()
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):#添加请求
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):#执行一步
        seqs, is_prefill = self.scheduler.schedule()#调度序列
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)#计算已调度token数
        token_ids = self.model_runner.run(seqs, is_prefill)#运行序列
        self.scheduler.postprocess(seqs, token_ids, is_prefill)#后处理序列
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]#获取输出
        return outputs, num_tokens#返回输出和已调度token数

    def is_finished(self):#判断是否完成
        return self.scheduler.is_finished()

    def cache_stats(self):#获取缓存统计信息
        return self.scheduler.block_manager.stats()

    def generate(
        self,#生成
        prompts: list[str] | list[list[int]],          #可以直接输入文本或者token id
        sampling_params: SamplingParams | list[SamplingParams],   #采样参数
        use_tqdm: bool = True,
    ) -> list[str]:#返回生成结果
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)#进度条
        if not isinstance(sampling_params, list): #如果采样参数不是列表，则将采样参数重复len(prompts)次
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):#将prompt和采样参数配对
            self.add_request(prompt, sp)#添加请求
        outputs = {}#输出
        prefill_throughput = decode_throughput = 0.#预填充和解码吞吐量
        step_count = 0#步数
        progress_interval = 16#进度间隔
        while not self.is_finished():#循环直到完成
            t = perf_counter()#记录当前时间
            output, num_tokens = self.step()#执行一步
            if num_tokens > 0:#如果已调度token 大于0，则计算预填充吞吐量
                prefill_throughput = num_tokens / (perf_counter() - t)#预填充吞吐量
            else:#如果已调度token 小于0，则计算解码吞吐量
                decode_throughput = -num_tokens / (perf_counter() - t)#解码吞吐量
            step_count += 1#步数加1
            if step_count % progress_interval == 0 or self.is_finished():#如果步数是进度间隔的倍数或者完成，则更新进度条
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })#更新进度条
            for seq_id, token_ids in output:#将输出配对
                outputs[seq_id] = token_ids
                pbar.update(1)#更新进度条
        pbar.close()#关闭进度条
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]#将输出排序
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]#将输出转换为文本
        return outputs#返回生成结果
