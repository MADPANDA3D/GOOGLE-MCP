# Compatibility matrix

The ordered 151-tool catalog in [tool-catalog.md](tool-catalog.md) is the
tool-by-tool compatibility matrix. The runtime manifest test projects and
freezes every row using native identity, aliases, input/output schemas,
annotations, and confirmation metadata.

| Contract | v1.0.2 | Current source |
|---|---|---|
| Native tools | 151 | 151 |
| Ordered identity SHA-256 | `156235e3f91fa345ae4e11308e20bddcd209822cc2cc1740e120dd6788cf52b6` | `156235e3f91fa345ae4e11308e20bddcd209822cc2cc1740e120dd6788cf52b6` |
| Full compatibility projection SHA-256 | `9f12a0b7bdc2df0b01ee1ecf6f8b3ff178b6b6bf56ad5ddab7f90be821b5b505` | `78f70e9968baeaf022d7d61a209a31dc967768748e72910231d78a440ef950a3` |
| Descriptor SHA-256 | `2c777ccf9f5528e8a3fcaea8de69535ca8a8aae8f85fa622fa55e7d76ffc76d0` | `26fd74c3fc50367ea4f4f8ae1f35ef09d43866f7cea085b852eaf120adb18487` |

The identical ordered identity hash proves that no native tool, canonical
identity, or alias is lost. The full projection changes because current source
adds decoded-byte pagination to the Gmail attachment input schema. The change
prevents content-page truncation without removing a tool identity or changing
risk and confirmation behavior.

Run the frozen contract check with:

```bash
uv run --frozen pytest fastmcp/tests/test_tool_manifest.py
```
