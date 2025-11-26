TITLE = """# 🏅 DABStep Leaderboard"""

INTRODUCTION_TEXT = """
The [Data Agent Benchmark for Multi-step Reasoning (DABStep)](https://huggingface.co/blog/dabstep) is looking to measure and push the state-of-the-art in Data Analysis by LLMs.
The benchmark is composed of ~450 data analysis questions ([Dataset Link](https://huggingface.co/datasets/adyen/data-agents-benchmark)) centered around 1 or more documents that agents will have to understand and cross reference in order to answer correctly.

We have set up a notebook to quickly get an agent baseline using the free Huggingface Inference API: [Colab Notebook](https://colab.research.google.com/drive/1pXi5ffBFNJQ5nn1111SnIfjfKCOlunxu)

Check out the official technical reports here:
[Adyen Report](https://www.adyen.com/knowledge-hub/data-agent-benchmark-for-multi-step-reasoning-dabstep)
[Hugging Face Report](https://huggingface.co/blog/dabstep)

Join the discussion on the [discord server!](https://discord.gg/8cCbjmCH4d)

Reproduce the baseline results with the agent code we open sourced [here](https://huggingface.co/spaces/adyen/DABstep/tree/main/baseline)
"""

SUBMISSION_TEXT = """
## Submissions
Scores are expressed as the percentage of correct answers. 

Each question calls for an answer that is either a string (one or a few words), a number, or a comma separated list of strings or floats, unless specified otherwise. There is only one correct answer. 
Hence, evaluation is done via quasi exact match between a model’s answer and the ground truth (up to some normalization that is tied to the “type” of the ground truth).


We expect submissions to be json-line files with the following format. 
Mandatory fields are: `task_id` and `agent_answer`. However, `reasoning_trace` is optional:
```
{"task_id": "task_id_1", "agent_answer": "Answer 1 from your agent", "reasoning_trace": "The different steps by which your model reached answer 1"}
{"task_id": "task_id_2", "agent_answer": "Answer 2 from your agent", "reasoning_trace": "The different steps by which your model reached answer 2"}
```

Our scoring function can be found [here](https://huggingface.co/spaces/adyen/data-agents-benchmark/blob/main/dabstep_benchmark/evaluation/scorer.py).
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""@misc{DABstep,
      title={DABstep: Data Agent Benchmark for Multi-step Reasoning}, 
      author={Alex Egg and Martin Iglesias Goyanes and Friso Kingma and Andreu Mora and Leandro von Werra and Thomas Wolf},
      year={2025},
      eprint={2506.23719},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.23719}
}"""


VALIDATION_GUIDELINES = """
## Benchmark Validation Standards

All submissions are initially added to the **Unvalidated Leaderboard**. The Adyen/Hugging Face team will attempt, with the participation of the respective submission team, to validate any entries that rank within the top 10.

**Validation** confirms that a submission's results were achieved using a novel approach involving data analysis agents. To support validation, participants must provide clear evidence of their methodology. This can be done in one of the following ways:

- **Preferred:** Share a research paper or blog post along with the source code to enable full reproducibility.
- Submit a complete dataset that includes **reasoning traces** demonstrating how the results were produced.
- Provide access to an **API** that the Adyen/Hugging Face team can use to independently validate and reproduce results.

Our goal with **DABStep** is to foster rapid progress and collaboration in the open research community. We strongly encourage participants to share their work and open-source their code whenever possible.

Once validated, submissions will be featured and showcased on the **Validated Leaderboard**, including annotations indicating the validation method used (e.g., `traces`, `code`, `API`).
"""
