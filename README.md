# Overview
This repository is solutions of [2023년 국립국어원 인공 지능 언어 능력 평가: 감정 분석 과제](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103&clCd=END_TASK&subMenuId=sub01) [2023년 국립국어원 인공 지능 언어 능력 평가: 감정 분석 과제](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103&clCd=END_TASK&subMenuId=sub01).

The solutions should be run in the GPU with 24GB. 
For the large llm, we need quantization as `gptq_convert.py`.

# Requirements
To train the model, libraries should be installed using
```
pip install --upgrade pip
git clone https://github.com/NVIDIA/apex
pip install -v --disable-pip-version-check --no-cache-dir ./apex
pip install transformers ipdb datasets peft bitsandbytes fire einops sentencepiece apache_beam
pip install deepspeed tensorboardX triton openpyxl
pip install --upgrade multiprocess
pip install texttable toml scipy
pip install auto-gptq optimum scikit-learn
pip install deepspeed==0.9.4  transformers==4.32.0
```

After install, deepspeed and transformers library should be modified.
Replace 411-th line of `PYTHON_LIB_PATH/dist-packages/transformers/trainer.py` with
```
if False: # _is_quantized_and_base_model and not _is_peft_model:
```
And replace 1039-th line of `PYTHON_LIB_PATH/dist-packages/deepspeed/runtime/engine.py` with
```
try:
  self.module.half()
except:
  pass
```
This will prevent some error caused by qlora.

# Results (Oct. 23, 2023)

Competition 1
![image](https://github.com/LearnItAnyway/corpus_ko_AI_competition_2023/assets/76693336/fc372709-5b19-4994-846a-2347c46b632e)


Competition 2
![image](https://github.com/LearnItAnyway/corpus_ko_AI_competition_2023/assets/76693336/7c8eb2f2-c6c3-47ab-a836-d31090c55e1a)


### The codes for each competition are in `comp_1_EA` and `comp_2_SC`, respectively.

