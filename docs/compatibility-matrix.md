# Compatibility matrix

The ordered 151-tool catalog in [tool-catalog.md](tool-catalog.md) is the
tool-by-tool compatibility matrix. The runtime manifest test projects and
freezes every row using native identity, aliases, input/output schemas,
annotations, and confirmation metadata.

| Contract | v1.0.2 | Current source |
|---|---|---|
| Native tools | 151 | 151 |
| Ordered identity SHA-256 | `156235e3f91fa345ae4e11308e20bddcd209822cc2cc1740e120dd6788cf52b6` | `156235e3f91fa345ae4e11308e20bddcd209822cc2cc1740e120dd6788cf52b6` |
| Full compatibility projection SHA-256 | `9f12a0b7bdc2df0b01ee1ecf6f8b3ff178b6b6bf56ad5ddab7f90be821b5b505` | `9285718449c46d25ec4bac61bd7d52dd32303422ad53091c10320c9689ed711d` |
| Descriptor SHA-256 | `2c777ccf9f5528e8a3fcaea8de69535ca8a8aae8f85fa622fa55e7d76ffc76d0` | `dea8318c59cef9c2e5343c2f6bfee7b7f23e2cea6e365747f2f9c13a21c20f88` |

The identical ordered identity hash proves that no native tool, canonical
identity, or alias is lost. The full projection changes because current source
adds decoded-byte pagination to the Gmail attachment input schema. The change
prevents content-page truncation without removing a tool identity or changing
risk and confirmation behavior.

Run the frozen contract check with:

```bash
uv run --frozen pytest fastmcp/tests/test_tool_manifest.py
```
