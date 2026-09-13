from pathlib import Path

p = Path("scripts/apply_feature_norm_phase2.py")
text = p.read_text()
old = '''replace(
    "src/agent.rs",
    "use crate::registry::LayerRegistry;\\n",
    "use crate::registry::LayerRegistry;\\nuse crate::layers::custom::feature_norm::DEFAULT_EPSILON as FEATURE_NORM_DEFAULT_EPSILON;\\n",
)
'''
new = '''replace(
    "src/agent.rs",
    "};\\nuse crate::registry::LayerRegistry;\\n\\nfn push_u32",
    "};\\nuse crate::registry::LayerRegistry;\\nuse crate::layers::custom::feature_norm::DEFAULT_EPSILON as FEATURE_NORM_DEFAULT_EPSILON;\\n\\nfn push_u32",
)
'''
if text.count(old) != 1:
    raise SystemExit(f"expected one generic Agent LayerRegistry import patch, found {text.count(old)}")
p.write_text(text.replace(old, new))
