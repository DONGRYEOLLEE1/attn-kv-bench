# TokenDock

## 초록

TokenDock은 대규모 언어모델 서빙에서 자주 언급되는 세 가지 최적화 요소인 Grouped-Query Attention, KV cache, 그리고 PagedAttention 계열의 block 기반 KV 메모리 관리가 실제 지연 시간과 처리량에 어떤 차이를 만드는지를 측정하기 위해 작성된 실험용 프로젝트다. 본 프로젝트는 답변 품질 평가를 의도적으로 제외하고, 서빙 경로의 기계적 성능만을 비교한다. Apple Silicon과 MPS 환경에서는 CUDA 커널에 강하게 의존하는 vLLM을 그대로 재현하기 어렵기 때문에, 본 레포는 toy dense transformer를 이용해 동일한 개념을 통제된 조건에서 검증한다. 실험 결과, GQA와 KV cache를 함께 사용한 경로는 baseline 대비 TTFT, decode throughput, end-to-end latency를 모두 개선했으며, block 기반 paged KV 관리 경로는 contiguous cache 대비 peak KV memory를 추가로 낮추는 효과를 보였다.

## 1. 배경

Autoregressive LLM 서빙에서 가장 반복적으로 등장하는 비용은 이전 토큰의 문맥을 다시 계산하는 비용과, 이전 문맥으로부터 만들어진 key-value 상태를 저장하는 비용이다. baseline 구현이 prefix 전체를 매 토큰마다 다시 계산하는 방식이라면, 문맥이 길어질수록 추론 지연은 빠르게 증가한다. 여기에 multi-turn 세션이 겹치면, 이전 턴의 문맥을 얼마나 효율적으로 재사용하느냐가 서비스 수준 지표를 크게 좌우하게 된다. 이 문제를 해결하는 대표적 기법이 KV cache이며, attention 구조 차원에서는 GQA가 KV head 수를 줄여 cache 부담을 완화하고, 시스템 구현 차원에서는 PagedAttention류의 block allocator가 KV 메모리 단편화와 over-allocation을 완화한다.

## 2. 목표

이 프로젝트의 목표는 하나의 거대한 모델을 잘 서빙하는 것이 아니라, 서빙 최적화 기법이 성능 지표에 미치는 영향을 재현 가능하게 보여주는 것이다. 이를 위해 baseline 경로로는 vanilla multi-head attention과 no-cache 경로를 사용하고, 최적화 경로로는 GQA와 persistent KV cache를 결합한 엔진, 그리고 여기에 block 단위 KV 메모리 관리 기법을 추가한 엔진을 사용했다. 최종 출력은 사람이 읽기 쉬운 JSON 파일과 PNG 시각화 자료이며, 어떤 multi-turn 질문 세트를 기준으로 측정했는지도 결과 안에 함께 포함한다.

## 3. 방법

직접 비교한 엔진은 세 가지다. 첫 번째는 `baseline_mha_no_cache`로, 모든 토큰 생성 단계에서 prefix 전체를 다시 계산하는 가장 단순한 방식이다. 두 번째는 `gqa_contiguous_cache`로, query head와 key-value head를 분리하는 GQA를 적용하고 이전 토큰의 KV를 연속 버퍼에 유지하는 방식이다. 세 번째는 `gqa_paged_cache`로, GQA와 persistent KV cache를 유지하면서도 KV 메모리를 고정 크기 block 단위로 관리하여 PagedAttention의 핵심 아이디어를 반영한 방식이다. 이 설계는 vLLM의 PagedAttention에서 직접 영감을 받았지만, 본 레포는 CUDA 커널 재현이 아니라 개념적 효과의 재현을 목표로 한다.

구현의 핵심 코드는 [engines.py](/Users/drlee/workspace/dev/tokendock/tokendock/engines.py), [benchmark.py](/Users/drlee/workspace/dev/tokendock/tokendock/benchmark.py), [workload.py](/Users/drlee/workspace/dev/tokendock/tokendock/workload.py), [plots.py](/Users/drlee/workspace/dev/tokendock/tokendock/plots.py)에 있다. 모델 설정은 [config.py](/Users/drlee/workspace/dev/tokendock/tokendock/config.py)에 정리돼 있으며, 실험에 사용한 toy dense transformer는 `hidden_size=256`, `layers=6`, `query heads=8`, `KV heads=2`, `max_seq_len=1024` 설정으로 동작한다.

## 4. 실험 환경

실험은 Apple Silicon / MPS 환경에서 실행되었다. 따라서 본 결과는 “CUDA 기반 production serving 절대 수치”가 아니라, 동일한 하드웨어와 동일한 toy 모델에서 최적화 기법 간의 상대 비교 결과로 해석해야 한다. 벤치마크는 warmup 이후 세 번의 측정 실행을 수행하고, 각 실행은 네 개의 multi-turn 세션을 포함한다. 한 세션은 일곱 개의 턴으로 구성되며, 각 턴마다 이전 문맥을 계속 유지하도록 설계되어 KV cache 재사용 효과가 드러나게 했다.

실행은 아래 두 명령으로 재현할 수 있다.

```bash
cd ~/workspace/dev/tokendock
uv sync
uv run src/benchmark.py
uv run src/plot_results.py
```

## 5. 워크로드

벤치마크는 단순한 한 문장 프롬프트가 아니라, 운영 로그 해석과 장기 문맥 유지를 요구하는 multi-turn 세션으로 구성된다. 예를 들어 수조 관찰 및 이벤트 추론 세션은 물살 변화, 그림자, 유리 진동, 먹이, 조명 변화 같은 사건을 여러 턴에 걸쳐 누적시키며, 후반부 질문은 앞선 모든 턴을 기억하고 응답해야 한다. 다른 세션들은 프로젝트 로그 요약, 공급망 이상 탐지, 시스템 지연 원인 분석과 같은 형태를 띤다. 전체 질문 원문은 [benchmark_results.json](/Users/drlee/workspace/dev/tokendock/results/benchmark_results.json)의 `workload.sessions` 항목에 그대로 포함되어 있다.

## 6. 정량 결과

최신 결과 JSON는 [benchmark_results.json](/Users/drlee/workspace/dev/tokendock/results/benchmark_results.json)에 저장된다. 현재 측정 기준으로 `gqa_contiguous_cache`는 baseline 대비 mean TTFT를 45.908% 줄였고, mean decode tokens/s를 56.548% 높였으며, mean total latency를 50.819% 줄였다. `gqa_paged_cache` 역시 baseline 대비 mean TTFT를 44.134% 줄였고, mean decode tokens/s를 50.504% 높였으며, mean total latency를 48.772% 줄였다. 추가로 `gqa_paged_cache`는 contiguous cache 대비 peak KV memory를 22.727% 감소시켰다. 즉, GQA와 KV cache는 직접적인 속도 이득을 만들었고, paged KV 관리 방식은 그 위에 메모리 효율을 개선하는 방향으로 작동했다.

## 7. 시각화

아래 그림은 baseline과 두 최적화 경로 사이의 평균 지연 시간과 처리량을 비교한다.

![Latency and Throughput](assets/latency_throughput.png)

그림 파일은 [latency_throughput.png](/Users/drlee/workspace/dev/tokendock/assets/latency_throughput.png)에 있다. 그림을 보면 두 최적화 경로 모두 baseline보다 mean TTFT와 end-to-end latency가 낮고, decode throughput은 더 높게 나타난다. 이는 prefix 재계산 제거와 GQA 도입이 직접적인 서빙 성능 향상으로 이어졌음을 보여준다.

아래 그림은 peak KV memory 사용량을 비교한다.

![KV Memory](assets/kv_memory.png)

그림 파일은 [kv_memory.png](/Users/drlee/workspace/dev/tokendock/assets/kv_memory.png)에 있다. contiguous cache와 비교했을 때 paged KV manager는 peak KV bytes를 더 낮게 유지한다. 다시 말해 PagedAttention-style block 관리의 주요 이점은 속도보다는 메모리 압박 완화에서 더 분명하게 관찰된다.

## 8. 논의

이 실험이 보여주는 가장 중요한 사실은, KV cache를 도입하지 않은 baseline은 문맥 길이가 길어질수록 prefix를 반복 계산하느라 지연과 처리량 모두에서 빠르게 불리해진다는 점이다. 반대로 GQA와 KV cache를 도입하면 attention 경로가 더 적은 KV head를 다루고, 이전 문맥을 반복 계산하지 않아도 되기 때문에 TTFT와 total latency가 함께 내려간다. 여기에 paged KV 관리까지 더하면 contiguous cache와 비교해 속도 차이는 크지 않을 수 있으나, 메모리 사용량을 더 촘촘하게 관리할 수 있어 장기적으로는 더 많은 동시 요청이나 더 긴 문맥에 유리한 방향을 만든다.

## 9. 한계

이 프로젝트는 공개 ~2B 모델을 직접 서빙해 얻은 결과가 아니다. 현재 머신이 Apple Silicon / MPS이기 때문에, CUDA 커널과 vLLM 런타임을 그대로 재현하는 것은 비현실적이었다. 따라서 본 실험은 “vLLM의 핵심 아이디어를 toy dense transformer 위에 구현해 상대 비교를 수행한 결과”로 읽는 것이 맞다. 그럼에도 불구하고 구조적으로 무엇이 속도 개선을 만들고, 무엇이 메모리 압박을 줄이는지는 충분히 분리해서 관찰할 수 있다.

## 10. 재현 산출물

최종 산출물은 세 가지다. 첫째, [benchmark_results.json](/Users/drlee/workspace/dev/tokendock/results/benchmark_results.json)은 벤치마크 환경, 모델 설정, 전체 multi-turn 세션 원문, 엔진별 요약 지표, 비교 지표, 최종 결론을 모두 포함한 구조화된 JSON이다. 둘째, [latency_throughput.png](/Users/drlee/workspace/dev/tokendock/assets/latency_throughput.png)은 속도 중심 비교를 위한 시각화다. 셋째, [kv_memory.png](/Users/drlee/workspace/dev/tokendock/assets/kv_memory.png)은 메모리 효율 비교를 위한 시각화다.

## 11. 참고 문헌

PagedAttention 및 vLLM의 핵심 아이디어는 `Efficient Memory Management for Large Language Model Serving with PagedAttention` 논문과 vLLM 블로그를 참고했다. GQA의 구조적 배경은 `GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints`를 참고했다.

- https://arxiv.org/abs/2309.06180  
- https://vllm.ai/blog/vllm  
- https://arxiv.org/abs/2305.13245
