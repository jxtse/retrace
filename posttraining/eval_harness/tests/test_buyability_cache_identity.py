"""Equivalent SMILES must preserve per-request identity on cache hits."""
import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from tools.check_buyability import CheckBuyabilityTool, _StockIndex, _build_sqlite_from_iter

RAW = 'N1CC[C@H](N)[C@H](OC)C1'
CANON = 'CO[C@@H]1CNCC[C@@H]1N'

@pytest.fixture
def tool(tmp_path):
    path = str(tmp_path / 'stock.sqlite')
    n = _build_sqlite_from_iter(path, [CANON, 'CCO'])
    t = CheckBuyabilityTool({})
    t._stocks = {'fixture': _StockIndex(path, count=n)}
    return t

@pytest.mark.parametrize('spellings', [(RAW, CANON, RAW), (CANON, RAW, CANON), ('OCC', 'CCO', 'OCC'), ('NCC', 'CCN', 'NCC')])
def test_single_cache_hit_echoes_current_input(tool, spellings):
    first = json.loads(tool.execute(smiles=spellings[0]))
    for spelling in spellings[1:]:
        row = json.loads(tool.execute(smiles=spelling))
        assert row['input_smiles'] == spelling
        assert row['canonical_smiles'] == first['canonical_smiles']
        assert row['buyable'] == first['buyable']
        assert row['found_in_stocks'] == first['found_in_stocks']
    assert len(tool._result_cache) == 1


def test_batch_rows_keep_requested_spelling_and_completion_join(tool):
    tool.execute(smiles_list=[RAW])
    requested = [CANON, RAW, 'CCO', 'OCC']
    out = json.loads(tool.execute(smiles_list=requested))
    assert [r['input_smiles'] for r in out['results']] == requested
    joined = {r['input_smiles']: r['buyable'] for r in out['results']}
    assert all(joined[s] for s in requested)
    assert out['n_buyable'] == len(requested)


def test_concurrent_warm_cache_keeps_response_identity(tool):
    tool.execute(smiles=RAW)
    requested = [RAW, CANON] * 8
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda s: json.loads(tool.execute(smiles=s)), requested))
    assert [r['input_smiles'] for r in rows] == requested
    assert all(r['buyable'] for r in rows)


def test_invalid_input_does_not_reuse_buyable_cache(tool):
    tool.execute(smiles=RAW)
    out = json.loads(tool.execute(smiles_list=[CANON, 'not-a-smiles']))
    assert out['results'][0]['buyable'] is True
    assert 'error' in out['results'][1]
    assert out['n_buyable'] == 1
