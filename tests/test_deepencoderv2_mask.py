from __future__ import annotations
# pyright: reportMissingImports=false, reportPossiblyUnboundVariable=false, reportAttributeAccessIssue=false

import unittest

try:
    import torch

    from dsv4_vision_vllm.deepencoderv2 import CustomQwen2Decoder

    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


def _tiny_decoder() -> CustomQwen2Decoder:
    torch.manual_seed(7)
    model = CustomQwen2Decoder(
        decoder_layer=1,
        max_position_embeddings=32,
        hidden_dimension=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=32,
        attn_implementation="eager",
    )
    return model.eval()


def _oracle_mask(
    token_types: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = token_types.shape[1]
    positions = torch.arange(seq_len, device=token_types.device)
    causal = positions[:, None] >= positions[None, :]
    image_to_image = (token_types[:, :, None] == 0) & (
        token_types[:, None, :] == 0
    )
    allowed = causal.unsqueeze(0) | image_to_image
    additive = torch.zeros(allowed.shape, dtype=dtype, device=token_types.device)
    additive = additive.masked_fill(~allowed, torch.finfo(dtype).min)
    return additive.unsqueeze(1), allowed


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires the pinned Anemll Torch runtime")
class DeepEncoderV2MaskTests(unittest.TestCase):
    def test_image_prefix_mask_matches_explicit_qwen2_oracle(self) -> None:
        model = _tiny_decoder()
        inputs = torch.randn(1, 4, 8)
        token_types = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        expected_mask, expected_allowed = _oracle_mask(token_types, inputs.dtype)
        seen_masks = []

        def capture_mask(_module, _args, kwargs):
            seen_masks.append(kwargs["attention_mask"].detach().clone())

        handle = model.model.layers[0].self_attn.register_forward_pre_hook(
            capture_mask, with_kwargs=True
        )
        with torch.no_grad():
            actual = model(
                inputs, token_types, use_cache=False
            ).last_hidden_state
        handle.remove()

        self.assertEqual(len(seen_masks), 1)
        torch.testing.assert_close(seen_masks[0].eq(0)[:, 0], expected_allowed)

        inner = model.model
        with torch.no_grad():
            reference = super(type(inner), inner).forward(
                inputs_embeds=inputs,
                attention_mask={"full_attention": expected_mask},
                use_cache=False,
            ).last_hidden_state
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_future_image_token_affects_earlier_image_output(self) -> None:
        """Image-prefix tokens use bidirectional attention, not causal attention."""
        model = _tiny_decoder()
        inputs = torch.randn(1, 4, 8)
        token_types = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

        changed = inputs.clone()
        changed[:, 1, :] += 3.0

        with torch.no_grad():
            baseline = model(inputs, token_types).last_hidden_state
            perturbed = model(changed, token_types).last_hidden_state

        delta = (baseline[:, 0, :] - perturbed[:, 0, :]).abs().max().item()
        self.assertGreater(
            delta,
            1e-5,
            "the first image token did not attend to the later image token; "
            "the custom bidirectional image mask was not applied",
        )


if __name__ == "__main__":
    unittest.main()
