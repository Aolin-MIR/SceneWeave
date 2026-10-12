"""Check portable snapshots without loading a model or calling a remote API."""
from pathlib import Path
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))

class PortableReleaseTests(unittest.TestCase):
    def test_actual_pinned_prompt_functions_load_without_upstream_network_code(self):
        from official_superwriter_adaptation import official_functions
        functions = official_functions()
        self.assertIn('fixture-query', functions['generate_prompt']('plan', use_query='fixture-query'))
        self.assertIn('fixture-plan', functions['generate_prompt']('write', previous_output='<answer>fixture-plan</answer>'))

    def test_all_data_checksums_preserve_the_original_full_bytes(self):
        spec = importlib.util.spec_from_file_location('restore_fixture', ROOT / 'tools/restore_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.restore(verify_only=True)
        self.assertGreater(result['verified_datasets'], 0)
        self.assertFalse(result['labels_or_partitions_changed'])

    def test_model_substitution_is_rejected_before_a_transport_call(self):
        spec = importlib.util.spec_from_file_location('transport_fixture', ROOT / 'tools/experimental_inference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.inference('fixture', 'System', 'Prompt', model='unauthorized-model')

if __name__ == '__main__':
    unittest.main()
