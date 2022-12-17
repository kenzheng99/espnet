# ESPnet ONNX Project
MIIS Capstone Project
Karthik Ganesan, Samarth Navali, Kenneth Zheng
Advisor: Shinji Watanabe


## Introduction
Taking ML models from research to production without friction is one of the major challenges in realizing the real-world value from research outcomes. In our project, we explore the case of tackling this challenge for the domain of modern end-to-end speech recognition models, building on top of the popular ESPnet speech toolkit developed and maintained by CMU. Currently industry practitioners (e.g. the AWS speech team) take models developed in toolkits like ESPnet and optimize them for various target applications and hardware platforms. Typically, this process requires a lot of knowledge in both the ML and systems domains, since it requires re-implementing models with APIs that can interface directly with the target hardware. The addtional effort required to optimize models for each target application (language like Java, C++, C#, and hardware like Android, Raspberry Pi, NVIDIA GPU, etc) increases the effort required to productionize research models.

Thus, having an automated way to export research models to be optimized for each target production environment is very valuable. For our project we focus on ONNX, one such community platform aiming to streamline the research-to-production pipeline, and explore integrating ONNX into the ESPnet toolkit to seamlessly export models to different production settings.

### Why ONNX?
ONNX (the Open Neural Network Exchange) is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common `.onnx` file format which represents a model as a computational graph using these operators to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

The two main advantages include:
1. **Interoperability**
ONNX allows researchers to develop in their preferred framework without worrying about downstream inference implications, and then convert models for use with any supported inference engine. For example, this should allow us to develop models in PyTorch using ESPnet and convert them to ONNX for downstream inference applications.
2. **Harware Optimization Access**
ONNX makes it easier to access hardware optimizations. Use ONNX-compatible runtimes and libraries designed to maximize performance across hardware.

Theoretically, ESPnet is built upon PyTorch, which is already compatible with ONNX and provides APIs to convert models from PyTorch to ONNX. However, we found that there are some limitations to this support, which necessitate writing model code in a specific way to make sure they are able to be exported. Our project focused on investigating these limitations and updating the ESPnet codebase to properly support the ONNX format.

---

## ONNX Tutorial
In this section, we write up our work on this project in a tutorial-style format, which we plan to add to the ESPnet documentation for future developers to understand how to export and debug ONNX models and write ONNX compatible code.

### PyTorch ONNX export function

Exporting any model into the ONNX format can be done using the `torch.onnx.export()` method. This method takes a torch model as input and then outputs an ONNX optimized model. This ONNX model can be used without torch and can be deployed on systems with various backends. To export a model completely which can be used properly for inference, one has to understand the inputs that the export method takes. This method takes in the following parameters as input:

- `model` - PyTorch model (`nn.Module`) to be exported to the ONNX format. Note that only the default `forward()` method is exported.
- `dummy_input` - a dummy input that is of the same shape as the input to the PyTorch model (e.g. a tuple that contains an entry for each input to the model's `forward()` function)
- `out_file_path` - path where the `.onnx` ONNX model file should be saved
- `verbose` - if True, prints some stats during the export
- `opset_version` - specifiying the ONNX operation set version, please check ONNX/PyTorch documentation for compatibility
- `input_names` - list of strings representing names to assign to each input parameter (corresponding to each argument passed to the model's `forward()`). These are arbitrary, and only used to pass in parameters by name when running the ONNX model.
- `output_names` - like above, a list of names to assign to each output parameter (corresponding to return values of the `forward()` function)
- `dynamic_axes` - this is the most important parameter for sequence-based models, allowing the specification of dimensions that can vary over different instances of input (e.g. speech inputs of different durations). This takes in a dictionary of dictionaries. Each key of the top level dictionary is an input or output name (from above). Each sub-dictionary has a key which is a dimension of the given input/output which can vary at inference time, and the value is an arbitary name for that dimension.

For example, here is the code to export an ESPnet encoder model:
```python
feats_dim = 80
length = 200
dummy_input = (torch.randn(1, length, feats_dim), torch.Tensor([length]))
torch.onnx.export(
    encoder,
    dummy_input,
    'encoder.onnx',
    verbose=False,
    opset_version=15,
    input_names=["feats", "feats_lens"],
    output_names=["encoder_out", "encoder_out_lens"],
    dynamic_axes={
        "feats": {1: "feats_length"},
        "encoder_out": {1: "enc_out_length"},
    },
)
```

In the above given example the `encoder` model is being exported and its being stored in the file `encoder.onnx`. The names of input nodes in the model are `feats` and `feats_lens` and the output nodes are `encoder_out` and `encoder_out_lens`. There are two dynamic axes specified - dimension 1 of the input `feats` (which is the time dimension of the speech features input), and dimension 1 of the output `encoder_out` (which is the time dimension of the output hidden states).

Internally, this export function uses `torch.jit.trace()`, which runs a tracing operation by running the dummy input through the model and recording all of the operations performed to get to the output. The key to writing ONNX compatible code is to understand this tracing process, since not all ways of implementing models in PyTorch can be properly traced. In the next section, we discuss some potential pitfalls and implementation issues that may not be properly traced.

### Considerations when writing ONNX compatible code

A couple of suggestions to have PyTorch code compatible with ONNX to get accurate expected output: 

1. In general, try to avoid branching (if/else) in the forward function. During the trace, only one branch of each if/else is taken for the given dummy input, and therefore only that branch is traced. Branching is okay only if the same branch will be taken for each instance of the function during inference, and is independent of the input contents.
2. In general, try to avoid using any variables that are not torch Tensors, as they will be treated as constants during the trace. For example, numpy arrays, python int/float/bool primitives, python lists, etc will all be treated as constants, so only use them if necessary for variables that will be constant regardless of the input.
3. If possible, try keeping the implementation for the inference pass separate from operations used only during training, as only the inference pass will be exported to ONNX.

There are a couple more niche patterns and details which are not ONNX-compatible, for more information check out the official [torch.onnx documentation](https://pytorch.org/docs/stable/onnx.html).

Note also that we have only discussed tracing in this section. There is an alternative way to export to ONNX using the torch scripting API instead (which can, for example, encode branching into the exported model), but this usually also requires code changes (potentially more affected code than with using tracing). A model compatible with both methods will also be more efficient when exported using the tracing method, so we decided not to investigate scripting for this project.

### The ESPnet_onnx project
The [ESPnet_onnx](https://github.com/espnet/espnet_onnx) project is a repository originally written and maintained by Masao Someki. It is a standalone codebase that is able to export and run many ESPnet models using the ONNX format and runtime. The codebase mainly consists of two modules
- The `espnet_onnx/export/` directory contains code and scripts to actually export ESPnet PyTorch models to the onnx format. For ASR models, the entry point is `export_asr.py` which has a function to take an ESPnet `Speech2Text` model and exports the encoder, decoder, ctc, and any other separable modules as `.onnx` files into the user's `~/.cache/espnet_onnx` directory.
- The `espnet_onnx/asr/` and `espnet_onnx/tts/` directories contain an inference pipeline which is able to run exported ASR/TTS models without using PyTorch at all, using only numpy array-based operations and the Python ONNXRuntime API.

While working on this project, we noticed that the `export/` directory contains a re-implementation of a lot of modules from ESPnet. We were initially confused as to why this was necessary, since theoretically it should be pretty simple to export PyTorch models to ONNX using the `torch.onnx.export()` function described above. However, the ESPnet codebase actually contains a number of issues that make certain modules incompatible with ONNX exporting. Masao's solution was to reimplement modules within the ESPnet_onnx codebase that were incompatible, which works but is not great since it introduces a lot of hacky fixes, code duplication, and overall poor maintainability. For our contributions, we chose to investigate the original code in ESPnet, and try to fix the ONNX compatibility issues natively at the source code.

### Case study: encoder
Now, we will walk through an example of our debugging process for finding and solving ONNX compatibility issues while exporting a specific model - the `ConformerEncoder` model, which was the first model we tried. We used this [pretrained model](https://huggingface.co/pyf98/librispeech_conformer/tree/main/exp/asr_train_asr_conformer8_raw_en_bpe5000_sp) as an example, which is a Conformer+CTC+Transformer model trained on Librispeech. Note that the encoder, decoder, CTC, and potentially other modules need to be exported separately, and the beam search decoding pipeline needs to be re-implemented as in the ESPnet_onnx codebase.

To start, we tried to simply export the ESPnet model's encoder without any modifications, using the `torch.onnx.export()` function exactly as described above. This successfully ran a trace and generated the `encoder.onnx` function. Then, we tried to test this encoder using features generated from a sample wav file, using something like the following code:
```python
import librosa
import onnxruntime as rt

# load example file and get features using ESPnet frontend
audio, sr = librosa.load('example.wav')
audio_lens = [len(audio)]
feats, feats_lens = espnet_model.asr_model.frontend(audio, audio_lens)

# run exported ONNX encoder
onnx_model = rt.InferenceSession('encoder.onnx')
onnx_enc_out, onnx_enc_out_lens = onnx_model.run(
    None, {"feats": feats.numpy(), "feats_lens": feats_lens.numpy()}
)
```

However, when testing this function on an audio file, we ran into an error indicating that the ONNX export didn't correctly capture the encoder's computational graph. Note that this error doesn't happen when we test with the dummy input, so it indicates that the issue is with some part of the computational graph not being dynamic enough to deal with different input lengths.
```bash!
Traceback (most recent call last):
  File "test_export.py", line 65, in <module>
    onnx_enc_out, onnx_enc_out_lens = onnx_model.run(None, {'feats': feats.numpy()})
  File "/Users/kentoshin/projects/espnet_onnx/tools/venv/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 200, in run
    return self._sess.run(output_names, input_feed, run_options)
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Where node. Name:'/encoders/encoders.0/self_attn/Where' Status Message: /Users/runner/work/1/s/onnxruntime/core/providers/cpu/math/element_wise_ops.h:503 void onnxruntime::BroadcastIterator::Init(ptrdiff_t, ptrdiff_t) axis == 1 || axis == largest was false. Attempting to broadcast an axis by a dimension other than 1. 67 by 249
```

This error gives us a node name (`/encoders/encoders.0/self_attn/Where`), so our next step was to visualize the actual computational graph and find the error. To do this, we can use a great online app called [Netron](https://netron.app/), which allows us upload a `.onnx` model file and visualize it as a computational graph. Using Netron, we can then search for the node where the error occured, and look at its context to try to figure out which lines of PyTorch code it corresponds to. Unfortunately, there is no easy automatic way to do this, it just requires knowledge of the model structure and ESPnet codebase. In our example shown below, we found that this `Where` node appeared right before a `Softmax` node, and from other context were able to match it to the attention masking operation in ESPnet's `attention.py`.

![](https://i.imgur.com/9xfKWpA.png)

This part of the code is in the `MultiHeadAttention` module, and is a standard operation that is
necessary for Transformer-based models to operate on batches with different length inputs in them. The `mask` variable is generated by the `make_pad_mask()` function, and is designed to mask out the zero-padding tokens appended to shorter examples to prevent the attention mechanism from attending to them. Below, we show the original implementation of `make_pad_mask()` in ESPnet:

```python=
def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """
    [some documentation omitted]
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask
```

This function has a number of issues that could potentially cause an incorrect trace during ONNX exporting, namely:
- Lots of `if/else` branching, which should be avoided as much as possible as the tracer will only take one branch for each conditional during the export process.
- Usage of non-Torch data types and operations which should also be avoided, such as line 13 (converts length to a Python list) and line 18 (uses Python's max operation and int casting). During export, any non-tensor values will be treated as a constant.
- The dynamic slicing operation in lines 35-37. This is the most critical issue that was directly causing the trace to fail, as it's simply not supported by the tracer.

We noticed that these complexities are mainly caused by the optional parameters `xs` and `length_dim`. Upon doing an audit of the ESPnet codebase, we found that these optional parameters are mostly only used in things like frontends, GlobalMVN, etc which do not need to support ONNX, while most calls to it in models like the Transformer/ConformerEncoder are only using the required `lengths` input. Therefore, we decided to refactor `make_pad_mask()` into two separate functions:

```python=
def make_pad_mask(lengths, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor): Batch of lengths (B,).

    Returns:
        BoolTensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    lengths = lengths.long()
    bs = lengths.size(0)
    if maxlen is None:
            maxlen = lengths.max()

    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask

def make_pad_mask_with_reference(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor with the shape of a reference tensor."""
    # this is equivalent to the legacy implementation from above (for now)
```

By removing the `xs` and `length_dim`, we are able to dramatically simplify this function into something that is perfectly ONNX compatible. We also kept the legacy implementation as `make_pad_mask_with_reference()`, for use in code that isn't required to be ONNX compatible. We believe it might be possible to incorporate those optional parameters into a single implementation instead of splitting it (e.g. ESPnet_onnx attempts this), but it is quite tricky to make a generic implementation, so we leave this possibility for future refactoring work.

One additional change that we did is to the `MultiHeadedAttention.forward_attention()` function (the original function which threw the ONNX error). Instead of using the `torch.masked_fill()` function as in the original implementation, we simply add a large negative value to masked tokens. These operations are equivalent before a softmax as the softmax will map large negative values to 0. The below image shows this change and the new computational graph after our changes to `make_pad_mask()` and `forward_attention()`:

![](https://i.imgur.com/DddNvOF.png)

As you can see, these two changes have dramatically simplified the computational graph, making it more readable, and solved the ONNX export error we originally found. In fact, these were the only two changes needed to successfully export the original `ConformerEncoder`. We have made a [PR into ESPnet](https://github.com/espnet/espnet/pull/4821) to make these changes. Once that is merged we can delete the corresponding re-implementations from ESPnet_onnx, which is a step is the right direction from ESPnet.


### Case study: decoder
Next, we will briefly discuss our changes required to get the `TransformerDecoder` exported as well. We used a similar testing process to the encoder from above, using the same pretrained model's decoder and trying to export it directly first.

Firstly, a minor change we needed to make was to "monkey-patch" the `forward()` function to replace it with `forward_one_step()`, which is the actual forward function used in inference for ESPnet models. 
```
decoder.forward = decoder.forward_one_step
```
Having the ```forward``` operation identical during the export and inference of the model is critical to get the right computational graph exported. Note that the `forward_one_step()` function only does the decoding for one token, so the full decoding pipeline with beam search needs to be re-implemented to support ONNX (like it already is in the ESPnet_onnx repository).

When exporting the decoder with this patch, we then got an error which had to do with the `cache` input to `forward_one_step()`. Below is the function header for `TransformerDecoder.forward_one_step()`:

```python
# espnet2/asr/decoder/transformer_decoder.py
def forward_one_step(
    self,
    tgt: torch.Tensor,
    tgt_mask: torch.Tensor,
    memory: torch.Tensor,
    cache: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Forward one step.
    Args:
        tgt: input token ids, int64 (batch, maxlen_out)
        tgt_mask: input token mask,  (batch, maxlen_out)
                  dtype=torch.uint8 in PyTorch 1.2-
                  dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        memory: encoded memory, float32  (batch, maxlen_in, feat)
        cache: cached output list of (batch, max_time_out-1, size)
    Returns:
        y, cache: NN output value and cache per `self.decoders`.
        y.shape` is (batch, maxlen_out, token)
    """
```

The `cache` input contains the cached output from previous steps of the decoder. This is normally a fixed length tensor containing decoder states for `max_time_out` timesteps, with a masking mechanism to only attend to previously decoded states. However, there is an issue with the very first decoding step, where the ESPnet implementation normally passes in `None` as the cache input. While this makes sense intuitively since there are no cached values during the first step, this creates an issue with the ONNX export as it adds an if/else branch:

Below is a scenario in [decoder_layer.py](https://github.com/karthik19967829/espnet/blob/d98750c1ea88faec62aae7f908fd320a71184e4c/espnet/nets/pytorch_backend/transformer/decoder_layer.py#L85) where if cache is None as a dummy input, the `else` block is not executed. This leads to an error when during the inference forward pass the else block execution is required.

```python
# espnet/nets/pytorch_backend/transformer/decoder_layer.py
if cache is None:
    tgt_q = tgt
    tgt_q_mask = tgt_mask
else:
    tgt_q = tgt[:, -1:, :]
    residual = residual[:, -1:, :]
    tgt_q_mask = None
    if tgt_mask is not None:
        tgt_q_mask = tgt_mask[:, -1:, :]
```
To solve this, we need a path of execution that is the same for the first step as well. We solve this by exporting the model with dummy input such that cache is never `None` as shown below.

```python
dummy_cache = [
    torch.zeros((1, 2, decoder.decoders[0].size))
    for _ in range(len(decoder.decoders))
]

```
As this adds a redundant initial zero vector to the cache, we need to then ignore this with a slicing operation in the transformer forward pass, like:

```python
# x is the concatenation of all previous timesteps stored in cache 
# along with the current timestep
x = torch.cat([cache, x], dim=1)
x = x[:, 1:, :]
```

With this change, the decoder is able to successfully export, and can run properly with the ESPnet_onnx inference pipeline. However, more changes may be needed in ESPnet, as the beam search module that calls the decoder needs to as well provide initial zeros to the initial state/cache. We propose making the following change to handle this, although we haven't fully tested it yet.

```python 
if states[0] is None:
    batch_state = [
        np.zeros((1, 1, self.odim), dtype=np.float32)
        for _ in range(self.n_layers)
    ]
else:       # transpose state of [batch, layer] into [layer, batch]
    batch_state = [
        np.concatenate([states[b][i][None, :] for b in range(n_batch)])
        for i in range(self.n_layers)
    ]
```

---

## Our Contributions 

As discussed above, we had to make several changes to make sure that the models could be directly exported from the espnet codebase and for inference as well. To ensure this we had to make changes in the ESPnet codebase and suggest changes in the stand alone ESPnet_onnx codebase as well. In the next couple of sections we discuss about the changes in both the codebases. The sample model that we used was a 12-block conformer model along with 6-block transformer + CTC model. This model was trained on the Librispeech dataset which is a 1000-hour audiobook dataset. The pretrained models are available [here](https://huggingface.co/pyf98/librispeech_conformer).

### ESPnet PR

![](https://i.imgur.com/bJHf74n.png)

We have created a [PR](https://github.com/espnet/espnet/pull/4821) to the main branch of ESPnet. In this PR we suggest the following changes, described in more details in the case study sections above.

- Changes to natively support ONNX exporting of ESPnet transformer-based encoders
    - re-implementation of `make_pad_mask`
    - changing `masked_fill` to an addition operation
    - minor changes to multi-head attention module
    
- Changes to natively support ONNX exporting of ESPnet transformer-based decoders
    - initialize the cache to zeros instead of `None` the decoder forward method
    - same changes as above to the multi-head attention module

### ESPnet_onnx PR


The standalone ESPnet_ONNX handles things in a different way. It has different implementations for each of the encoder and calls each of the specific implementation of the model during the exportation. Due to the numerous implementations, there were a lot of `if/else` blocks in the code which could have been avoided. Anothe issue was that the code was not standardized as each implementation had a varying number of parameters. We proposed to solve all these issues with the following PR.

![](https://i.imgur.com/nlfsK8q.png)

In this PR we tackle several issues,
- We propose the creation of a standard wrapper class `EncoderWrapper` for all the encoders. The standard implementation helps in returning a standard dummy input, input names and output names for the model. With this change we are also able to get rid of the numerous `if/else` blocks for different models/modules in the code during exportation.
- We delete of all the now-redundant re-implementations that are natively supported to clean the repository
- We add a example script to export a single model so as to help anyone who would want to export any model of their choice
- We add a script that tests any espnet model (exported or normal) end-2-end. This was crucial in testing the inference correctness of our models during development, and we believe it might be helpful for future developers.
- We add a script to benchmark the difference in inference speed between the original ESPnet and the ONNX exported model versions.

We believe that with these changes, anyone who wants to export models into the onnx format will have enough templates which would make their process easier.

### Benchmark and results 
We ran a simple benchmark to compare the inference latency of a PyTorch and ONNX version of an ASR model on CPU.
Model: [Conformer+Transformer+CTC ASR model](https://huggingface.co/pyf98/librispeech_conformer).
Dataset : [Dummy subset](https://huggingface.co/datasets/patrickvonplaten/librispeech_asr_dummy) of the Librispeech dataset
Hardware: Laptop CPU (Macbook Pro 16" with a 6-core Intel i7@2.6Ghz)

| Model Platform | Avg Latency (s) |
| -------------- | --------------- |
| PyTorch        | 15.6            |
| ONNX           | 4.37            |

We achieve **3.57x** speedup with our natively exported ONNX model as compared to the original ESPnet PyTorch model as shown above.

### Conclusion and future work

We were able to deepdive into the ESPnet and ESPnet_onnx codebases to get a deeper understanding of their internal implementation details and limitations, and learned about the ONNX framework and export process works. We focused on reducing redundant effort between the two codebases by providing the ability to export models directly from ESPnet. We propose our changes through the PRs mentioned, and with this allow transformer based encoders and decoders to be natively ONNX compatible, which would significantly improve the maintainability of the ESPnet_onnx project and allow many researchers to more easily deploy their models to production.

In the future, we plan to more thoroughly test our changes and merge them into the ESPnet project. Then, we plan to test and debug all other types of encoders, decoders, and other model types within ESPnet to make all of them natively ONNX export-friendly. We also plan to integrate a CI (continuous integration) test into the repository, for future developers to automatically test if their model implementations are ONNX compatible. Finally, we plan to adapt this document as a tutorial within the current ESPnet documentation to help anyone who wants to understand the ONNX framework and exporting/debugging process in the context of ESPnet. 


### Acknowledgements
We would like to acknowledge Masao Someki for his work in developing ESPnet_onnx and helping us understand and debug ONNX-related issues. We would also like to acknowledge Xuankai Chang and Prof. Shinji Watanabe for providing advising and support throughout this project.