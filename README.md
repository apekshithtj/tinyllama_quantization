\# tinyllama quantization



I wanted to understand what actually breaks when you quantize an LLM

to INT4. So I took TinyLlama 1.1B, ran it at FP16, INT8 and INT4, and

then went layer by layer to see which parts of the network are most

responsible for the quality drop. The interesting part wasn't really

the headline numbers, it was the per-layer sweep.



Runs on a free Colab T4.



\## what's in here

src/                   reusable functions

loader.py            HF model + dataset loaders

quantize.py          per-layer INT4 swaps via bitsandbytes

evaluate.py          ppl, throughput, peak memory

sensitivity.py       layer-by-layer and block-by-block sweeps

activations.py       forward hooks for activation analysis

plotting.py          matplotlib helpers

scripts/               entry points, one per experiment

tests/                 pytest unit tests

results/               json + png outputs



\## setup



```bash

git clone https://github.com/apekshithtj/tinyllama\_quantization.git

cd tinyllama\_quantization

pip install -r requirements.txt

```



Needs a CUDA GPU because bitsandbytes won't run on CPU. On Colab:

Runtime > Change runtime type > T4 GPU.



\## running



Four scripts, in order:



```bash

python -m scripts.run\_compare\_variants     # \~5 min: FP16 vs INT8 vs INT4

python -m scripts.run\_layer\_sensitivity    # \~30-40 min: per-layer sweep

python -m scripts.run\_block\_sensitivity    # \~60-90 min: attention vs MLP

python -m scripts.run\_activation\_analysis  # \~5 min: forward hooks

```



Each writes a json and a png into results/. The two sensitivity sweeps

take a while because each runs perplexity 22 or 44 times.



\## results



All numbers below are from a clean Colab T4 run. Raw json in `results/`,

plots in `results/figures/`.



\### headline comparison



| Variant | Perplexity | Tokens/sec | Peak memory |

| ------- | ---------- | ---------- | ----------- |

| FP16    | 8.31       | 35.3       | 3.51 GB     |

| INT8    | 8.35       | 9.0        | 2.61 GB     |

| INT4    | 8.68       | 158.1      | 2.18 GB     |



INT8 is basically free on quality (ppl moves from 8.31 to 8.35, which

is within run-to-run noise) but the bitsandbytes INT8 matmul kernels

are actually slower than plain FP16 tensor cores on a T4 at this model

size. That surprised me at first but it's a known thing.



INT4 costs about 4.5% in perplexity and saves another \~0.5 GB on top

of INT8. The big throughput jump at INT4 (158 tok/s) is the optimized

4-bit matmul kernels finally beating FP16.



!\[Pareto plot](results/figures/pareto.png)



\### per-layer sensitivity



For each decoder layer, I quantized only that layer to INT4 and left

everything else in FP16, then measured perplexity.



!\[Layer sensitivity](results/figures/layer\_sensitivity.png)



The thing that jumped out: at TinyLlama scale, no single layer is

catastrophic. Even the worst layer only adds about 0.04 perplexity.

The top 5 sensitive ones are 21, 16, 2, 7, 18.



Layer 21 being the worst makes sense to me. It's the last decoder layer

before lm\_head, so any error there feeds directly into the output

projection.



\### attention vs MLP



Same sweep but finer: for each layer, I quantized only the attention

block (q/k/v/o) or only the MLP block (gate/up/down).



!\[Block sensitivity](results/figures/block\_sensitivity.png)



On average attention and MLP are within noise of each other

(7.4007 vs 7.4043, baseline 7.3954). I expected a clearer answer here

based on what's in the LLM.int8() and AWQ papers, but it didn't show

up at this scale.



What did show up: layer 21's MLP is the worst single block in the

whole model at 7.4287, well above any other block. The pattern is

much messier than "attention is always more sensitive" or vice versa.



\### activation distributions



Forward hooks on six layers (early, middle, late, both attention

output and MLP output) during a single forward pass.



The most interesting bit was watching how activation magnitudes grow

with depth, especially in the MLP down projection:



| layer                    | max abs | std    |

| ------------------------ | ------- | ------ |

| layers.0.mlp.down\_proj   | 0.25    | 0.009  |

| layers.10.mlp.down\_proj  | 0.55    | 0.040  |

| layers.20.mlp.down\_proj  | 5.44    | 0.262  |



That's about 20x growth in standard deviation from layer 0 to layer 20.

This lines up with the outlier story from the LLM.int8() and AWQ

papers: layers with heavier tails force the INT4 quantization grid

wider, which wastes precision on the typical values near zero. It also

explains why layer 21's MLP was the worst block in the sweep above.



INT4 and FP16 activation stats come out within \~2% of each other for

the same layer, which makes sense: bitsandbytes dequantizes weights

back to FP16 before the matmul, so the activations are almost the

same shape regardless of weight precision.



\## tests



```bash

pytest tests/

```



The quantization tests need CUDA and skip themselves otherwise. The

one I used most during development was `test\_int4\_close\_to\_fp16`,

which copies weights from an FP16 layer to a new layer, quantizes

the new one to INT4, runs both on the same input, and asserts the

MSE between the outputs is small. If you ever break the quantization

pipeline this is the test that fires first.



\## design notes



\*\*bitsandbytes, not GPTQ.\*\* I tried GPTQ first but it's a whole-model

algorithm and the layer-by-layer sweep gets awkward against that.

bitsandbytes Linear4bit lets you swap individual modules, which is

what made the sensitivity analysis possible. The bnb INT4 numbers are

slightly worse than GPTQ in absolute terms but the shape of the

findings doesn't change.



\*\*TinyLlama, not Llama-2 7B.\*\* T4 has 15 GB. TinyLlama in FP16 fits

twice over, which I need for the sensitivity sweep (clone the model,

quantize one layer, measure, throw away). 7B would force offloading

and the sweep would take days.



\*\*WikiText-2 for perplexity.\*\* Standard benchmark for this scale,

so the absolute numbers are comparable to what papers report.



\*\*lm\_head is not quantized.\*\* Common recipe across QLoRA and the

GPTQ defaults. Quantizing the output projection costs more perplexity

than the memory saves.



\## caveats



The sweeps use 8 perplexity chunks per measurement instead of the

full test set, otherwise the per-layer loop would run for hours.

Enough to rank layers reliably, not a publication-quality absolute

number. You can bump n\_chunks to None in the scripts for a full sweep.



Activation analysis runs on a single prompt. Enough to see the shape

of each layer's distribution, not the tails. You'd want hundreds of

prompts for that.



Throughput is on greedy generation of 128 tokens. Real-world latency

with KV cache and longer outputs looks different. The numbers here

are useful for comparing variants but not for predicting deployed

inference cost.



\## things i'd do with more time



Implement activation-aware scaling like AWQ does. My own data shows

the magnitude growth that motivates AWQ, so it would be a natural

extension and I'd expect it to recover most of the INT4 perplexity gap.



Try it on a 7B model. The "attention vs MLP" story might be sharper

at larger scale, given that papers report it clearly at 7B+ and I

couldn't pull it out at 1.1B.



Replace bitsandbytes INT4 with GPTQ for the headline comparison

specifically. The absolute numbers would be slightly cleaner and

comparable to published baselines.



\## troubleshooting



\*\*CUDA not available on Colab:\*\* you didn't set the runtime to GPU.



\*\*Out of memory mid-sweep:\*\* each iteration deepcopies the model,

which temporarily doubles VRAM. If you're on a smaller GPU than T4,

drop max\_length in evaluate.py from 2048 to 1024.



\*\*bitsandbytes complaining about libcudart:\*\* the wheel needs to

match your CUDA version. `pip install -U bitsandbytes` and read its

diagnostic output.



\*\*Perplexity higher than expected:\*\* most often this is because you

measured the baseline with a different n\_chunks than the sweep.

Always use the same value.



\*\*Throughput numbers jumping between runs on Colab:\*\* the GPU is

shared. Run 3 times, take the median.



\## license



MIT

