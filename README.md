# tinyllama quantization

A small project where I take TinyLlama 1.1B, quantize it down to INT8
and INT4, and look at what breaks. The interesting part isn't really
the headline FP16 vs INT4 numbers, it's the layer-by-layer sweep
that shows which parts of the network are actually responsible for
the quality drop.

Runs on a free Colab T4.

## what's in here

```
src/                   reusable functions
  loader.py            HF model + dataset loaders
  quantize.py          per-layer INT4 swaps via bitsandbytes
  evaluate.py          ppl, throughput, peak memory
  sensitivity.py       layer-by-layer and block-by-block sweeps
  activations.py       forward hooks for activation analysis
  plotting.py          matplotlib helpers

scripts/               entry points, one per experiment
  run_compare_variants.py
  run_layer_sensitivity.py
  run_block_sensitivity.py
  run_activation_analysis.py

tests/                 pytest unit tests
results/               json + png outputs land here
```

## setup

You need a CUDA GPU. A free Colab T4 is enough. Without CUDA the
bitsandbytes quantization just won't run.

```bash
git clone https://github.com/YOUR_HANDLE/tinyllama_quantization.git
cd tinyllama_quantization
pip install -r requirements.txt
```

On Colab: Runtime > Change runtime type > T4 GPU.

## running

Four scripts, intended to be run in order:

```bash
# headline FP16 vs INT8 vs INT4
python -m scripts.run_compare_variants

# which decoder layer is most sensitive to INT4?
python -m scripts.run_layer_sensitivity

# inside each layer, is attention or MLP the more sensitive half?
python -m scripts.run_block_sensitivity

# why are sensitive layers sensitive? look at the activation distributions
python -m scripts.run_activation_analysis
```

Each one writes a json with the numbers and a png plot into results/.

The compare script is fast, the two sweeps take a while (the sensitivity
ones run ppl 22-44 times each). On a T4 you're looking at maybe 30-50
minutes for the layer sweep and roughly double for the block sweep.

## what the experiments actually do

**compare variants.** Load the model three times (FP16, INT8 via
bitsandbytes, INT4 via bitsandbytes NF4), measure ppl + throughput +
peak memory, save a Pareto plot.

**layer sensitivity.** For each of the 22 decoder layers in TinyLlama:
clone the FP16 baseline, swap every linear inside that one layer for
INT4, leave all the other layers alone, measure ppl. The output is a
ranked list of "which layer matters most".

**block sensitivity.** Same idea, finer. Within each layer there are
two natural blocks: the attention block (q, k, v, o projections) and
the MLP block (gate, up, down projections). The script quantizes each
block in isolation and produces a grouped bar chart.

**activation analysis.** Registers forward hooks on a handful of layers,
runs the same prompt through both the FP16 and INT4 model, plots
histograms of the captured activations. This is where you can see
*why* some layers are sensitive: distributions with heavy tails force
the INT4 quantization grid wide, which destroys precision on the
typical values.

## tests

```bash
pytest tests/
```

The quantization tests need CUDA and skip themselves otherwise. The
evaluate tests run on CPU.

The one I actually used during development is `test_int4_close_to_fp16`.
It copies weights from an FP16 layer to a new layer, quantizes the new
one to INT4, runs both on the same input, and asserts the MSE is small.
If you ever break the quantization pipeline this is the test that fires
first.

## results

The scripts dump:

```
results/
  variants.json
  layer_sensitivity.json
  block_sensitivity.json
  figures/
    pareto.png
    layer_sensitivity.png
    block_sensitivity.png
    activations_fp16.png
    activations_int4.png
```

Once you've actually run it, the numbers from variants.json drop into
a table like:
| FP16    | 8.31       | 35.3       | 3.51 GB     |
| INT8    | 8.35       | 9.0        | 2.61 GB     |
| INT4    | 8.68       | 158.1      | 2.18 GB     |

(filled in with my actual numbers when you run on your machine. the
absolute values depend on hardware so they'll be slightly different
from anyone else's, but the shape of the trade-off is the same.)

## design notes

**bitsandbytes, not GPTQ.** I started with GPTQ but the layer-by-layer
sweep is awkward against a whole-model algorithm. bitsandbytes
Linear4bit lets you swap individual modules, which is what makes
sensitivity analysis straightforward. The bnb numbers are slightly
worse than GPTQ in absolute terms but the shape of the conclusions
doesn't change.

**TinyLlama, not Llama-2 7B.** T4 has 15 GB. TinyLlama in FP16 fits
twice over (we need a deep copy for the sensitivity sweep). 7B would
force offloading and the sweep would take days.

**WikiText-2 for ppl.** Standard for this scale of model, comparable
to published numbers.

**lm_head is not quantized.** Common recipe: quantizing the output
projection costs much more ppl than the memory saves. Most production
INT4 recipes skip it.

## caveats

- The sweeps use 8 ppl chunks per measurement. Enough to rank layers
  reliably, not a publication-quality absolute number. Bump n_chunks
  to None in the scripts for the full thing.
- Activation analysis runs on a single prompt. Enough to see the
  *shape* of a distribution, not the tails. Would want hundreds of
  prompts for that.
- Throughput is on greedy generation of 128 tokens. Real-world latency
  with KV cache + longer outputs looks different.

## troubleshooting

CUDA not available on Colab: you forgot to switch the runtime.

OOM during the sweep: each iteration deepcopies the model, which
temporarily doubles VRAM. On a smaller GPU than T4, drop max_length
from 2048 to 1024 in evaluate.py.

bitsandbytes complaining about libcudart: the wheel needs to match
your CUDA. On Colab this usually just works. Locally:
`pip install -U bitsandbytes` and read its diagnostic output.

ppl looks much higher than expected: most often this is because you
measured the baseline with a different n_chunks than the sweep.
Always use the same value.

throughput numbers jumping around between runs: on Colab the GPU is
shared. Run 3x and take the median.

## license

MIT, see LICENSE.
