# Balanced 10,000-record writer release

The writer release contains 10,000 TRAIN records and the fixed 160-record DEV. Each of six task types has 1,666 or 1,667 TRAIN records. Original record provenance and automatic admission status are retained. No training on this release is claimed.

The full text delivery, including the existing evaluator data and auxiliary supervision, is available to authorized collaborators at [the research data release](https://github.com/Aolin-MIR/script-generator-deploy/releases/tag/writer10000-evaluator-data-20261016). Source-specific text redistribution permissions have not been cleared for a new public release; this directory publishes the inventory only.

Topic variety is encouraged through varied TRAIN works and optional editorial briefs; a brief is not a verified genre label. See `manifest.json` for file counts and checksums.

## Version 2: additional human script preferences

The delivery now includes 123 additional human preference pairs from 34 connected task groups. The expanded evaluator file has TRAIN3123, DEV83 and TEST10; the original TRAIN3000 snapshot and separate increment are retained for provenance. Preserve task-group weights and human preference distributions; do not count the increment twice. No new training is claimed. Download `sceneweave_writer10000_evaluator_data_v2.zip` from the research release linked above.
