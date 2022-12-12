import argparse

import librosa
import numpy as np
import onnxruntime as rt
import torch

from espnet2.tasks.asr import ASRTask
from espnet2.utils import config_argparse
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.scorers.length_bonus import LengthBonus

def get_decoder_dummy_inputs(enc_size, decoder):
    tgt = torch.LongTensor([0, 0]).unsqueeze(0)
    enc_out = torch.randn(1, 100, enc_size)
    cache = [
        torch.zeros((1, 2, decoder.decoders[0].size))
        for _ in range(len(decoder.decoders))
    ]
    # pdb.set_trace()
    mask = torch.BoolTensor(1,2,2)
    return (tgt, mask, enc_out, cache)

def get_dynamic_axes(decoder):
    ret = {
            'tgt': {
                0: 'tgt_batch',
                1: 'tgt_length'
            },
            'tgt_mask': {
                0: 'tgt_mask_batch',
                1: 'tgt_mask_length',
                2: 'tgt_mask_length',
            },
            'memory': {
                0: 'memory_batch',
                1: 'memory_length'
            }
        }
    ret.update({
        'cache_%d' % d: {
            0: 'cache_%d_batch' % d,
            1: 'cache_%d_length' % d
        }
        for d in range(len(decoder.decoders))
    })
    return ret

def export_encoder(encoder, feats_dim, out_filename="encoder.onnx", test_input=None):
    """
    Export an encoder module to ONNX.

    Args:
        encoder (nn.Module): Encoder module to export
        feats_dim (int): Input feature dimension for the encoder
        out_filename (str): Filename to save ONNX model to
        test_input (tuple of (feats, feats_lens)):
            An optional test input to verify the ONNX model.
            If not given, test with the same dummy input used to export

    Currently tested: conformer
    """

    dummy_feats = torch.randn(1, 200, feats_dim)
    dummy_feats_lens = (
        torch.ones(dummy_feats[:, :, 0].shape).sum(dim=-1).type(torch.long)
    )
    dummy_input = (dummy_feats, dummy_feats_lens)

    torch.onnx.export(
        encoder,
        dummy_input,
        out_filename,
        verbose=False,
        opset_version=15,
        input_names=["feats", "feats_lens"],
        output_names=["encoder_out", "encoder_out_lens"],
        dynamic_axes={
            "feats": {1: "feats_length"},
            "encoder_out": {1: "enc_out_length"},
        },
    )

    # test ONNX encoder and compare outputs with PyTorch encoder
    if not test_input:
        test_input = dummy_input

    test_feats, test_feats_lens = test_input
    enc_out, enc_out_lens, _ = encoder(test_feats, test_feats_lens)
    enc_out = enc_out.detach().numpy()

    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_model = rt.InferenceSession(out_filename, options)
    onnx_enc_out, onnx_enc_out_lens = onnx_model.run(
        None, {"feats": test_feats.numpy(), "feats_lens": test_feats_lens.numpy()}
    )

    assert np.array_equal(enc_out.shape, onnx_enc_out.shape)
    assert np.allclose(enc_out, onnx_enc_out, atol=1e-5)
    mse = ((enc_out - onnx_enc_out) ** 2).mean()
    print(f"encoder successfully exported to {out_filename}, mse={mse}")


def export_decoder(asr_model, enc_out_size, out_name="decoder.onnx", test_input=None):
    
    print("exporting decoder now")
    decoder = asr_model.decoder
    decoder.forward = decoder.forward_one_step
    ctc_weight = asr_model.ctc_weight
    scorers = {}
    device = "cpu"

    ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
    token_list = asr_model.token_list
    scorers.update(
        decoder=decoder,
        ctc=ctc,
        length_bonus=LengthBonus(len(token_list)),
    )

    beam_search_transducer = None
    hugging_face_model = None
    hugging_face_linear_in = None
    lm_weight = 1.0
    ngram_weight = 0.9
    penalty = 0.0
    beam_size = 20
    dtype="float32"
    maxlenratio = 0.0
    minlenratio = 0.0

    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        ngram=ngram_weight,
        length_bonus=penalty,
    )

    beam_search = BeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=asr_model.sos,
        eos=asr_model.eos,
        vocab_size=len(token_list),
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "full",
    )

    beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
    # pdb.set_trace()

    # nbest_hyps = beam_search(x=torch.tensor(enc), maxlenratio=maxlenratio, minlenratio=minlenratio)

    decoder_inputs = ['tgt', 'tgt_mask', 'memory'] \
            + ['cache_%d' % i for i in range(len(decoder.decoders))]
    
    decoder_outputs = ['y'] + ['out_cache_%d' % i for i in range(len(decoder.decoders))]

    dummy_input_decoder = get_decoder_dummy_inputs(enc_out_size, decoder)

    dynamic_axes = get_dynamic_axes(decoder)

    torch.onnx.export(
        decoder,
        dummy_input_decoder,
        out_name,
        verbose=False,
        opset_version=15,
        input_names=decoder_inputs,
        output_names=decoder_outputs,
        dynamic_axes=dynamic_axes,
    )

    print("exported onnx decoder")



def export_ctc(ctc, enc_out_dim, out_filename="ctc.onnx", test_input=None):
    """
    Export a CTC module to ONNX.

    Args:
        ctc (nn.Module): CTC module to export
        enc_out_dim (int): Input hidden dimension for the CTC module
        out_filename (str): Filename to save ONNX model to
        test_input (tuple of (ctc_input,)):
            An optional test input to verify the ONNX model.
            If not given, test with the dummy input used to export

    """
    dummy_enc_out = torch.randn(1, 100, enc_out_dim)
    dummy_input = (dummy_enc_out,)
    ctc.forward = ctc.log_softmax
    torch.onnx.export(
        ctc,
        dummy_input,
        "ctc.onnx",
        verbose=False,
        opset_version=15,
        input_names=["x"],
        output_names=["ctc_out"],
        dynamic_axes={"x": {1: "ctc_in_length"}, "ctc_out": {1: "ctc_out_length"}},
    )

    # test ONNX CTC and compare outputs with PyTorch CTC
    if not test_input:
        test_input = dummy_input

    ctc_out = ctc.log_softmax(dummy_enc_out)
    ctc_out = ctc_out.detach().numpy()

    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_ctc = rt.InferenceSession(out_filename, options)
    onnx_ctc_out = onnx_ctc.run(None, {"x": dummy_enc_out.numpy()})[0]
    assert np.array_equal(ctc_out.shape, onnx_ctc_out.shape)
    assert np.allclose(ctc_out, onnx_ctc_out, atol=1e-5)
    mse = ((ctc_out - onnx_ctc_out) ** 2).mean()
    print(f"CTC successfully exported to {out_filename}, mse={mse}")


def export(asr_train_config: str, asr_model_file: str, test_wav: str):
    """Exports relevant modules from an ESPnet ASRModel to ONNX."""

    # load espnet model
    device = "cpu"
    asr_model, asr_train_args = ASRTask.build_model_from_file(
        asr_train_config, asr_model_file, device
    )
    asr_model.eval()

    # test encoder on one wav file
    audio, sr = librosa.load(test_wav)

    # get features using espnet frontend
    audio_tensor = torch.Tensor(audio).unsqueeze(0)
    audio_lengths = torch.Tensor([len(audio)])
    feats, feats_lens = asr_model.frontend(audio_tensor, audio_lengths)
    feats_lens = feats_lens.long()

    # export onnx encoder
    # export_encoder(
    #     asr_model.encoder,
    #     feats_dim=feats.shape[-1],
    #     test_input=(feats, feats_lens),
    # )

    # export CTC
    # export_ctc(
    #     asr_model.ctc,
    #     enc_out_dim=asr_model.encoder.encoders[0].size,
    # )

    # export decoder
    export_decoder(
        asr_model,
        asr_model.encoder.encoders[0].size,
    )


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR ONNX export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config",
        type=str,
        help="ASR training configuration",
        default="../../egs2/librispeech/asr1/exp/pyf98/librispeech_conformer/exp/asr_train_asr_conformer8_raw_en_bpe5000_sp/config.yaml",
    )
    group.add_argument(
        "--asr_model_file",
        type=str,
        help="ASR model parameter file",
        default="../../egs2/librispeech/asr1/exp/pyf98/librispeech_conformer/exp/asr_train_asr_conformer8_raw_en_bpe5000_sp/valid.acc.ave.pth",
    )
    group.add_argument(
        "--test_wav",
        type=str,
        help="wav file",
        default="../../test_utils/ctc_align_test.wav",
    )

    return parser


def main(cmd=None):
    # print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    export(**kwargs)


if __name__ == "__main__":
    main()