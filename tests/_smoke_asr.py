"""Quick smoke test for ASR aux loss with v2 cosine-similarity loss."""
import torch
from transbench.modules.asr_module import ASRAttentionMixin

# v2: default noise_std=0.3, cosine similarity loss, detach_target=False
# Old (low noise) should produce small loss; new default should be meaningful.

m_low = ASRAttentionMixin(hidden_size=64, num_attention_heads=4, noise_std=0.01)
m_new = ASRAttentionMixin(hidden_size=64, num_attention_heads=4)  # uses new default 0.3

m_low.train(); m_new.train()

torch.manual_seed(42)
x = torch.randn(1, 8, 64)

torch.manual_seed(0)
_ = m_low(x)
loss_low = m_low.asr_aux_loss()

torch.manual_seed(0)
_ = m_new(x)
loss_new = m_new.asr_aux_loss()

print(f"ASR loss (noise_std=0.01): {loss_low.item():.4e}")
print(f"ASR loss (noise_std=0.3):  {loss_new.item():.4e}")
print(f"Ratio (new/low): {loss_new.item() / max(loss_low.item(), 1e-30):.1f}x")

assert loss_new.item() > loss_low.item(), "Higher noise_std should produce larger cosine loss"
assert loss_new.item() > 0.001, "v2 ASR loss should be meaningful (> 0.001)"
print("ASR SMOKE TEST PASSED")
