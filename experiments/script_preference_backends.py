"""Preserve official pretrained heads while adapting complete script comparisons."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from torch import nn
from train_adaptation_reward import LowRankLinear

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party/strands-decider'
HYBRID_TARGETS = ('q_proj', 'v_proj', 'in_proj_qkv', 'in_proj_z',
                  'in_proj_a', 'in_proj_b', 'out_proj')


def official_decider():
    """Import the pinned upstream implementation without changing its source."""
    sys.path.insert(0, str(VENDOR / 'src'))
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel, load_head_state
    return StrandsDeciderConfig, StrandsDeciderModel, load_head_state


def encode_pair(tokenizer, row, reverse=False, backend='scalar'):
    """Keep the entire shared context and both candidate texts without truncation."""
    from train_contextual_script_comparator import pair_text
    text = pair_text(row, reverse)
    if backend != 'decider':
        return tokenizer(text, return_tensors='pt', truncation=False)
    official_decider()
    from strands_decider.schema import ChoiceQuestion
    from strands_decider.prompting import build_prompt
    from strands_decider.data.collate import SystemOneCollator
    question = ChoiceQuestion(instructions='Which candidate better fulfills the writing request with literary and dramatic quality?',
        criteria={'A': 'Candidate A is preferable', 'B': 'Candidate B is preferable'})
    prompt, rendered = build_prompt(text, question)
    enc = tokenizer(prompt, return_tensors='pt', truncation=False, return_offsets_mapping=True)
    offsets = enc.pop('offset_mapping')[0].tolist()
    indices = SystemOneCollator.option_token_index(offsets, rendered.option_spans,
                                                  len(prompt) - len(rendered.text))
    enc['opt_idx'] = torch.tensor([indices], dtype=torch.long)
    enc['n_slots'] = torch.tensor([2], dtype=torch.long)
    return enc


class PreservedFloat32Score(nn.Linear):
    """Train the existing reward readout in FP32 without discarding its weights."""
    def __init__(self, original):
        super().__init__(original.in_features, original.out_features,
            bias=original.bias is not None, device=original.weight.device, dtype=torch.float32)
        with torch.no_grad():
            self.weight.copy_(original.weight.float())
            if self.bias is not None:
                self.bias.copy_(original.bias.float())

    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.weight, self.bias)


class TextComparisonHead(nn.Module):
    """Attach a new scalar task head to the correctly loaded Qwen3.5 text decoder."""
    def __init__(self, torso):
        super().__init__()
        self.torso = torso
        self.config = torso.config
        self.score = nn.Linear(self.config.hidden_size, 1, bias=False, dtype=torch.float32)
        nn.init.normal_(self.score.weight, std=.001)

    def gradient_checkpointing_enable(self, **kwargs):
        self.torso.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids, attention_mask):
        hidden = self.torso(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True).last_hidden_state
        positions = attention_mask.sum(-1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), positions]
        return SimpleNamespace(logits=self.score(pooled.float()))


class DecisionComparisonHead(nn.Module):
    """Use the native pretrained pointer head; expose its two-option log odds."""
    def __init__(self, model):
        super().__init__()
        self.decision = model
        self.config = model.torso.config

    def gradient_checkpointing_enable(self, **kwargs):
        self.decision.torso.gradient_checkpointing_enable(**kwargs)

    def forward(self, **inputs):
        # Upstream calibration came from other tasks and is not transferred here.
        output = self.decision(**inputs, temperature=1.0)
        logits = output['logits']
        return SimpleNamespace(logits=(logits[:, 0] - logits[:, 1]).unsqueeze(-1))


def check_qwen_weights(torso, folder):
    """Verify representative text tensors against the actual downloaded checkpoint."""
    from safetensors import safe_open
    index = json.loads((Path(folder) / 'model.safetensors.index.json').read_text())['weight_map']
    parameters = dict(torso.named_parameters())
    checked = []
    for name in ('norm.weight', 'layers.0.mlp.down_proj.weight'):
        source = 'model.language_model.' + name
        with safe_open(str(Path(folder) / index[source]), framework='pt', device='cpu') as file:
            expected = file.get_tensor(source)
        if not torch.equal(parameters[name].detach().cpu(), expected):
            raise ValueError('Text decoder weight mapping differs: ' + name)
        checked.append(name)
    return checked


def build_backend(model_path, backend='scalar'):
    """Load each released architecture and retain its actual pretrained function."""
    from transformers import AutoModelForSequenceClassification, AutoConfig, Qwen3_5ForCausalLM
    if backend == 'decider':
        Config, Model, load_head = official_decider()
        config_file=Path(model_path)/'strands_decider_config.json'
        if not config_file.exists():config_file=Path(model_path)/'hobson_config.json'
        config = Config.from_json(str(config_file))
        config.base_model = str(ROOT / 'models/qwen3_5_2b_base')
        config.base_revision = None
        torso = Model._load_torso(config, None, 'sdpa')
        checked = check_qwen_weights(torso, config.base_model)
        from peft import PeftModel
        from transformers import AutoTokenizer
        torso = PeftModel.from_pretrained(torso, str(Path(model_path) / 'lora'), is_trainable=True,
                                          local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        decision = Model(config, torso, tokenizer)
        decision.head.load_state_dict(load_head(str(model_path)))
        model = DecisionComparisonHead(decision).cuda()
        model.backend_details = {'pretrained_head_preserved': True, 'native_lora_rank': config.lora_r,
            'native_lora_alpha': config.lora_alpha, 'targets': config.lora_targets,
            'base_model_revision': json.loads((Path(config.base_model)/'source_revision.json').read_text())['revision'],
            'pretrained_text_tensor_checks': checked, 'temperature': 1.0,
            'published_temperature_not_assumed_calibrated_for_scripts': True}
    else:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        hybrid = config.model_type == 'qwen3_5'
        if hybrid:
            lm = Qwen3_5ForCausalLM.from_pretrained(model_path, config=config.get_text_config(),
                dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa')
            torso = lm.model
            checked = check_qwen_weights(torso, model_path)
            del lm
            model = TextComparisonHead(torso).cuda()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path,
                dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa').cuda()
            model.score = PreservedFloat32Score(model.score)
            checked = []
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        targets = HYBRID_TARGETS if hybrid else ('q_proj', 'v_proj')
        adapted = []
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name.rsplit('.',1)[-1] in targets:
                parent, child = name.rsplit('.',1)
                setattr(model.get_submodule(parent), child, LowRankLinear(module, rank=8))
                adapted.append(name)
        model.score.requires_grad_(True)
        if not adapted:
            raise ValueError('No intended LoRA modules found')
        model.backend_details = {'rank':8,'alpha':16,'targets':list(targets),'adapted_modules':adapted,
            'pretrained_head_preserved':not hybrid,'new_task_head':hybrid,
            'pretrained_text_tensor_checks':checked}
    model.config.use_cache = False
    model.backend_details['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.backend_details['total_parameters'] = sum(p.numel() for p in model.parameters())
    return model
