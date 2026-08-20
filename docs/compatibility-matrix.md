# Compatibility matrix

The ordered 151-tool catalog in [tool-catalog.md](tool-catalog.md) is the
tool-by-tool compatibility matrix. The runtime manifest test projects and
freezes every row using native identity, aliases, input/output schemas,
annotations, and confirmation metadata.

| Contract | Existing deployment | Current source |
|---|---|---|
| Native tools | 151 | 151 |
| Ordered identity SHA-256 | `156235e3f91fa345ae4e11308e20bddcd209822cc2cc1740e120dd6788cf52b6` | `156235e3f91fa345ae4e11308e20bddcd209822cc2cc1740e120dd6788cf52b6` |
| Full compatibility projection SHA-256 | `b3b699ae930b92d9d59a61275bc3932fd60b695c15d36ef0cee31fd24dc92803` | `9f12a0b7bdc2df0b01ee1ecf6f8b3ff178b6b6bf56ad5ddab7f90be821b5b505` |
| Descriptor SHA-256 | `0951e1f3e30add5208730acbcee60f91fc8732148ff83951dacfc2d9ae2eb790` | `2c777ccf9f5528e8a3fcaea8de69535ca8a8aae8f85fa622fa55e7d76ffc76d0` |

The identical ordered identity hash proves that no native tool, canonical
identity, or alias is lost. The full projection changes because current source
adds explicit confirmation contracts, restricts the legacy raw request to
bounded reads, makes provider diagnostics opt-in, and corrects risk metadata.
Those changes tighten invocation safety without removing a tool identity.

Run the frozen contract check with:

```bash
uv run --frozen pytest fastmcp/tests/test_tool_manifest.py
```
