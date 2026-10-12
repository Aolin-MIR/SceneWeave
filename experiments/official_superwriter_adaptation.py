"""Load only pinned official SuperWriter pure prompt functions."""
import ast
import hashlib
import json
from pathlib import Path
REVISION = '436149d87908a462b5f74270cb43d797163f6e17'
SNAPSHOT = Path(__file__).resolve().parents[1] / "third_party/superwriter" / REVISION

def official_functions():
    """Load only audited pure functions, never upstream network code."""
    source = SNAPSHOT / 'Inference/superwrite_gen.py'
    provenance = json.loads((SNAPSHOT / 'provenance.json').read_text())
    expected = next(row['sha256'] for row in provenance['files'] if row['path'] == 'Inference/superwrite_gen.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    tree = ast.parse(source.read_text())
    names = {'extract_sections', 'generate_prompt'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    assert not any(isinstance(node, forbidden) for fn in nodes for node in ast.walk(fn))
    scope = {'__builtins__': {'len': len}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope
