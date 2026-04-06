---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1181
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: show me A blue "π" symbol and an arrow pointing to a smaller "3"
  sentences:
  - '**SCENE:** A dimly lit room with a man intently examining a brain on a table,
    surrounded by various medical equipment and tools.


    **OBJECTS:** Brain, table, medical equipment, tools, chair, headphones, wires,
    glasses


    **ACTIVITY:** The man is carefully dissecting the brain, using a scalpel to make
    precise cuts and examine its internal structures.


    **ANOMALY:** None'
  - '**SCENE:** A man in a hard hat and gray jacket stands in front of a concrete
    wall, with a wooden structure visible in the background.


    **OBJECTS:** Hard hat, gray jacket, concrete wall, wooden structure


    **ACTIVITY:** The man is likely a construction worker or inspector, as indicated
    by his attire and the presence of a concrete wall and wooden structure in the
    background.


    **ANOMALY:** None'
  - '**SCENE:** A black background with various mathematical symbols and a brain.


    **OBJECTS:** A blue "π" symbol, a thought bubble containing the number 3, an arrow
    pointing to a smaller "3", a white "3" in a box, and a gray brain.


    **ACTIVITY:** The image appears to be a visual representation of a mathematical
    concept or equation, possibly related to pi or the number 3.


    **ANOMALY:** None.'
- source_sentence: '**SCENE:** A dimly lit laboratory or workshop with a cluttered
    workbench and various equipment in th'
  sentences:
  - '**SCENE:** A dimly lit laboratory or workshop with a cluttered workbench and
    various equipment in the background, creating an atmosphere of intense focus and
    experimentation.


    **OBJECTS:** Desk lamp, computer monitor, keyboard, mouse, wires, cables, and
    a person wearing a white lab coat.


    **ACTIVITY:** A man with glasses and a gray hoodie is intently looking at something
    on the computer screen, possibly analyzing data or working on a project.


    **ANOMALY:** None.'
  - 'SCENE: A city street with a bus stop, a statue, and a car parked behind a fence.
    The atmosphere is urban and possibly touristy, with a mix of old and new architecture.
    OBJECTS: bus, statue, car, fence, flowers, sign, trees ACTIVITY: The bus is stopped
    at the bus stop, and people are walking around. ANOMALY: none'
  - 'SCENE: A close-up of a snake''s head and body, with a blurred green background.


    OBJECTS: snake, leaves, blurry green background.


    ACTIVITY: The snake is coiled up and appears to be looking at something.


    ANOMALY: none'
- source_sentence: A serene forest clearing with a large tree as the central feature,
    surrounded by lush greenery and a grassy field
  sentences:
  - 'SCENE: A close-up of a green snake wrapped around a red flower, with a blurred
    background suggesting a natural setting.


    OBJECTS: snake, flower, leaves


    ACTIVITY: The snake is coiled around the flower, its body visible in the foreground.


    ANOMALY: none'
  - 'SCENE: A sloth is hanging from a tree branch, surrounded by lush greenery and
    a blurred background. The atmosphere is serene and natural, with the sloth appearing
    relaxed and content.


    OBJECTS: sloth, tree branch, greenery, blurry background


    ACTIVITY: The sloth is hanging from the tree branch, likely resting or feeding.


    ANOMALY: None'
  - '**SCENE:** A serene forest clearing with a large tree as the central feature,
    surrounded by lush greenery and a grassy field.


    **OBJECTS:** The tree, a cave, rocks, and a large rock.


    **ACTIVITY:** The scene appears to be a still image, with no activity taking place.


    **ANOMALY:** None.'
- source_sentence: A simple, dark gray background with a cartoonish blue "π" symbol
    on the left side and three white "3"s in boxes on the right side
  sentences:
  - 'SCENE: A blurry, out-of-focus background with a green hue, suggesting a natural
    setting such as a forest or jungle. The atmosphere is calm and serene, with a
    sense of tranquility.


    OBJECTS: A large green leaf, a light-colored branch, and a small, furry creature
    with a long tail and claws.


    ACTIVITY: The creature is grasping the branch with its claws, possibly preparing
    to climb or hang from it.


    ANOMALY: None.'
  - '**Scene:** A simple, dark gray background with a cartoonish blue "π" symbol on
    the left side and three white "3"s in boxes on the right side.


    **Objects:** The blue "π" symbol and the three white "3"s in boxes.


    **Activity:** There is no activity depicted in the image.


    **Anomaly:** None.'
  - '**SCENE:** A close-up view of a tree trunk with green vines growing on it, set
    against a blurred background.


    **OBJECTS:** The tree trunk, green vines, and a watermark in the bottom-right
    corner.


    **ACTIVITY:** The image appears to be a still from a video or photograph showcasing
    the natural beauty of the tree and its surroundings.


    **ANOMALY:** None.'
- source_sentence: ' The person is working on a project, possibly building or repairing
    an electronic device.'
  sentences:
  - '**SCENE:** A futuristic, high-tech setting with a focus on advanced technology
    and machinery. The atmosphere is one of intense energy and power, with a sense
    of urgency and activity.


    **OBJECTS:** Large cylindrical structures, likely reactors or engines, with complex
    systems and components visible. The structures are made of metal and have a metallic
    sheen to them.


    **ACTIVITY:** The reactors or engines are emitting a bright, intense light, suggesting
    that they are operational and generating a significant amount of energy.


    **ANOMALY:** None apparent.'
  - 'SCENE: A close-up view of a green frog clinging to a tree trunk, with moss and
    leaves visible in the background.


    OBJECTS: tree trunk, moss, leaves, frog


    ACTIVITY: The frog is clinging to the tree trunk, possibly hunting for insects
    or resting.


    ANOMALY: None'
  - '**SCENE:** A person with long hair and a white shirt is sitting at a table with
    various electronic components, surrounded by a colorful and futuristic-looking
    environment.


    **OBJECTS:** The table has a variety of electronic components, including wires,
    circuit boards, and small devices. There are also some plants visible in the background.


    **ACTIVITY:** The person is working on a project, possibly building or repairing
    an electronic device.


    **ANOMALY:** None.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    ' The person is working on a project, possibly building or repairing an electronic device.',
    '**SCENE:** A person with long hair and a white shirt is sitting at a table with various electronic components, surrounded by a colorful and futuristic-looking environment.\n\n**OBJECTS:** The table has a variety of electronic components, including wires, circuit boards, and small devices. There are also some plants visible in the background.\n\n**ACTIVITY:** The person is working on a project, possibly building or repairing an electronic device.\n\n**ANOMALY:** None.',
    '**SCENE:** A futuristic, high-tech setting with a focus on advanced technology and machinery. The atmosphere is one of intense energy and power, with a sense of urgency and activity.\n\n**OBJECTS:** Large cylindrical structures, likely reactors or engines, with complex systems and components visible. The structures are made of metal and have a metallic sheen to them.\n\n**ACTIVITY:** The reactors or engines are emitting a bright, intense light, suggesting that they are operational and generating a significant amount of energy.\n\n**ANOMALY:** None apparent.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,181 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 4 tokens</li><li>mean: 18.12 tokens</li><li>max: 70 tokens</li></ul> | <ul><li>min: 38 tokens</li><li>mean: 99.89 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  |:--------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>**SCENE:** A dimly lit laboratory or workshop with a cluttered workbench and various equipment in th</code>         | <code>**SCENE:** A dimly lit laboratory or workshop with a cluttered workbench and various equipment in the background, creating an atmosphere of intense focus and experimentation.<br><br>**OBJECTS:** Desk lamp, computer monitor, keyboard, mouse, wires, cables, and a person wearing a white lab coat.<br><br>**ACTIVITY:** A man with glasses and a gray hoodie is intently looking at something on the computer screen, possibly analyzing data or working on a project.<br><br>**ANOMALY:** None.</code> |
  | <code>A serene beach scene with a palm tree in the foreground, a cloudy sky, and a body of water in the background</code> | <code>**SCENE:** A serene beach scene with a palm tree in the foreground, a cloudy sky, and a body of water in the background.<br><br>**OBJECTS:** Palm tree, leaves, sand, water, sky, clouds, and a logo in the bottom-right corner.<br><br>**ACTIVITY:** The image appears to be a still photograph, capturing a moment in time rather than depicting an ongoing activity.<br><br>**ANOMALY:** None.</code>                                                                                                    |
  | <code>A bus is parked at a bus stop.</code>                                                                               | <code>SCENE: A suburban street with a bus stop and a fence in front of a row of trees. The sky is overcast.<br><br>OBJECTS: bus, fence, streetlight, tree, car, grass, sidewalk, road, street sign, logo<br><br>ACTIVITY: A bus is parked at a bus stop.<br><br>ANOMALY: none</code>                                                                                                                                                                                                                              |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.11.0
- Sentence Transformers: 4.1.0
- Transformers: 4.57.1
- PyTorch: 2.10.0+cpu
- Accelerate: 1.10.1
- Datasets: 4.2.0
- Tokenizers: 0.22.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->