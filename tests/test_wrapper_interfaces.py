# pyright: reportMissingImports=false, reportPossiblyUnboundVariable=false
import unittest

try:
    from dsv4_vision_vllm.model import DeepseekV4VisionForCausalLM
    from vllm.model_executor.models.interfaces import supports_eagle3, supports_multimodal

    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires the pinned Anemll vLLM runtime")
class VisionWrapperInterfaceTests(unittest.TestCase):
    def test_wrapper_exposes_anemll_dspark_interface(self) -> None:
        self.assertTrue(supports_multimodal(DeepseekV4VisionForCausalLM))
        self.assertTrue(
            supports_eagle3(DeepseekV4VisionForCausalLM),
            "Anemll DSpark requires the target wrapper to satisfy SupportsEagle3",
        )
        self.assertTrue(
            hasattr(DeepseekV4VisionForCausalLM, "set_aux_hidden_state_layers")
        )
        self.assertTrue(
            hasattr(
                DeepseekV4VisionForCausalLM,
                "get_eagle3_default_aux_hidden_state_layers",
            )
        )


if __name__ == "__main__":
    unittest.main()
