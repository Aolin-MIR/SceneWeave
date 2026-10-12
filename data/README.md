# Prepared data

`catalog.json` is the complete upload inventory. Every `.gz` file decompresses to the exact original prepared JSON or JSONL bytes. Original labels, source identifiers, and split assignments are retained. Human supervision, synthetic teacher preferences, reference scenes, writer targets, and target-free requests have separate roles.

## Writer records

Writer JSONL records retain their task, original split, source group, inputs, role-ordered messages, target hashes, and provenance. Supported tasks include `adapt_scene`, `revise_text`, `continue_script`, `revise_with_feedback`, `write_scene_from_synopsis`, and `adaptation_process`. Process stages distinguish scene writing from full-script realization and revision. The released outer request template is shared by training and inference.

`writer/train.jsonl.gz` contains the prepared 3,391-row known-snapshot clean base. Its manifest explicitly records incomplete current protection and no new training authorization. `historical_train.jsonl.gz` contains the earlier actual 3,397-row release; it remains historical data and is not another independent dataset. Existing adapters may retain exposure to removed records.

Candidates under `writer/candidates/` are not automatically training-ready. `route_requests.jsonl.gz` contains source/brief inputs only. No generated text, inferred quality label, common initialization, or route preference is invented for those requests.

## Preferences

`preferences/comparator_training_data.json.gz` preserves the complete released dictionary and original TRAIN/DEV/TEST. Each comparison includes its writing request, two actual texts, preference probability, weight, dataset, and provenance. The naming of text slots does not convert a soft preference or tie into a hard winner. Synthetic labels remain synthetic, with their actual origin. Human targets are not smoothed or replaced during packaging.

The prepared training release has 3,000 pairs, including 269 human comparisons and 2,731 synthetic comparisons. DEV and TEST remain independent of training configuration selection. This repository does not publish a newly qualified trained checkpoint or claim dramatic-quality improvement from that count.

## Sources and auxiliary supervision

`prose_contexts.json.gz` contains original bounded passages with fixed work/author partitions. `aligned_adaptation_scenes.json.gz` contains verified prose-to-reference correspondence; a reference target is not proof that readers prefer that adaptation. Expert revisions, observed operations, HANNA dimensions, and external expert judgments retain their original purpose. Multiple judgments of one story are correlated observations, not independent stories.

## Source terms

This is a private research snapshot of prepared data the project already holds. Source-specific attribution and access conditions remain in record provenance and the catalog. There is no blanket data redistribution license. Before making a public data release, review source permissions at the work and dataset level; where necessary release metadata and acquisition scripts instead of copyrighted prose or screenplay excerpts. No source text is relicensed by the software repository.
