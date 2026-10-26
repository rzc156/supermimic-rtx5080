# Legacy: step-normalized benchmark JSONs

These are the original benchmark outputs whose denominator was the **executed
number of steps** (`norm_return = sum_rewards / (0.1 * executed_steps)`).
When a clip terminated early, the denominator shrank and the failed clip could
receive an inflated return.

They are kept for history only and are **superseded** by
`../benchmark_ckpt_*_fullclip*.json`, which use the full clip length `T-1` and
exclude the extra terminal reward step. Do not use these JSONs for conclusions.
