# Multi-Domain Verbose Instruction Generator

## Overview

An **Enhanced ReAct (Reasoning + Acting) agent** that generates detailed, step-by-step instructions for multi-domain data analysis and workflow tasks, matching the format of `verbose_instruction.txt` files in the [Spider2-V dataset](https://github.com/xlang-ai/Spider2-V).

**Input**: Task description (natural language instruction)
**Output**: Verbose, tutorial-style step-by-step instructions
**Domains Supported**: 11 domains (BigQuery, dbt, Airbyte, Airflow, Dagster, Excel, Jupyter, Metabase, ServiceNow, Snowflake, Superset)

---

## Features

### 1. Multi-Domain Support
- **11 Domains**: Automatically detects and generates instructions for:
  - BigQuery (data warehouse queries)
  - dbt (data transformation)
  - Airbyte (data integration)
  - Airflow (workflow orchestration)
  - Dagster (data orchestration)
  - Excel (spreadsheet operations)
  - Jupyter (notebook-based analysis)
  - Metabase (business intelligence)
  - ServiceNow (IT service management)
  - Snowflake (data warehouse)
  - Apache Superset (data visualization)

### 2. Few-Shot Learning
- **23 curated examples** across all domains
- Domain-specific instruction patterns and best practices
- Automatically selects relevant examples based on task type

### 3. Schema-Aware Generation (BigQuery)
- Integrates with TablesAnalyzer for BigQuery schema information
- Includes column names, types, and sample data in prompts
- Generates accurate SQL queries with actual column references

### 4. Flexible Processing Modes
- **Sequential processing**: Process N examples in order
- **Per-domain processing**: Process N examples per domain/task type
- **Single example preview**: Test with detailed output preview

### 5. Domain Detection
- Automatic keyword-based domain detection
- Scores each domain based on task description and table patterns
- Falls back to generic patterns when needed

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  INPUT: spider2-v-extended-dataset                  │
│  ├── airbyte/{instance_id}/{instance_id}.json       │
│  ├── airflow/{instance_id}/{instance_id}.json       │
│  ├── bigquery/{instance_id}/{instance_id}.json      │
│  └── ... (11 domains total)                         │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  MULTI-DOMAIN GENERATOR                             │
│  ┌───────────────────────────────────────────────┐  │
│  │ 1. Domain Detection (keyword scoring)         │  │
│  │ 2. Task Classification (LLM-based)            │  │
│  │ 3. Schema Loading (BigQuery only)             │  │
│  │ 4. Few-Shot Selection (domain-specific)       │  │
│  │ 5. Instruction Generation (Azure OpenAI o3)   │  │
│  └───────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  OUTPUT: verbose_instructions/                      │
│  ├── airbyte/{instance_id}/verbose_instruction.txt  │
│  ├── airflow/{instance_id}/verbose_instruction.txt  │
│  ├── bigquery/{instance_id}/verbose_instruction.txt │
│  └── ... (domain-grouped structure)                 │
└─────────────────────────────────────────────────────┘
```

---

## Directory Structure

### Input Structure
```
spider2-v-extended-dataset/
├── airbyte/
│   └── {instance_id}/
│       ├── {instance_id}.json          # Task metadata
│       └── verbose_instruction.txt     # Reference instruction
├── airflow/
├── bigquery/
└── ... (11 domains)
```

### Output Structure
```
verbose_instructions/
├── airbyte/
│   └── {instance_id}/
│       └── verbose_instruction.txt     # Generated instruction
├── airflow/
│   └── {instance_id}/
│       └── verbose_instruction.txt
├── bigquery/
│   └── {instance_id}/
│       └── verbose_instruction.txt
└── ... (domain-grouped)
```

---

## Usage

### Installation

```bash
# Install dependencies
pip install openai python-dotenv google-cloud-bigquery

# Set up environment variables
export AZURE_OPENAI_API_KEY="your-api-key"
```

### Command-Line Interface

```bash
# Test with a single example (shows preview)
python react_verbose_simple.py --num 1

# Process 10 examples sequentially
python react_verbose_simple.py --num 10

# Process 2 examples per domain (recommended for testing)
python react_verbose_simple.py --per-task-type 2

# Process 5 examples per domain
python react_verbose_simple.py --per-task-type 5

# Process all 494 examples
python react_verbose_simple.py --all
```

### Example Output

**Command:**
```bash
python react_verbose_simple.py --num 1
```

**Output:**
```
Initialized Enhanced Multi-Domain VerboseInstructionGenerator
  - Loaded 23 few-shot examples across all 11 domains
  - Supported domains: bigquery, dbt, airbyte, airflow, dagster,
                       excel, jupyter, metabase, servicenow,
                       snowflake, superset
  - Schema-aware generation: Enabled (BigQuery)

Processing 1 out of 494 examples...
================================================================================

[1/1] Processing: 56bc1d01-8790-4683-8670-856c498e5097 (domain: airbyte)
Question: I have transferred some data to a local sqlite DB using Airbyte...
Generating verbose instruction...
  - Detecting domain...
    Domain: airbyte
  - Classifying task type...
    Task type: code_based_querying
    Complexity: simple
  - Generating verbose instructions...
Saved verbose instruction to: ./verbose_instructions/airbyte/56bc1d01-8790-4683-8670-856c498e5097/verbose_instruction.txt

✓ Complete! Verbose instruction saved
  Total lines: 25

Successfully processed: 1/1
```

---

## Dataset Statistics

### spider2-v-extended-dataset Coverage

| Domain      | Examples | Description                      |
|-------------|----------|----------------------------------|
| BigQuery    | 40       | Data warehouse SQL queries       |
| Excel       | 62       | Spreadsheet operations           |
| ServiceNow  | 58       | IT service management            |
| Airbyte     | 48       | Data integration pipelines       |
| Metabase    | 48       | Business intelligence            |
| Jupyter     | 44       | Notebook-based data analysis     |
| Snowflake   | 44       | Data warehouse operations        |
| dbt         | 40       | Data transformation workflows    |
| Dagster     | 40       | Data orchestration               |
| Airflow     | 38       | Workflow orchestration           |
| Superset    | 32       | Data visualization               |
| **Total**   | **494**  | **11 domains**                   |

---

## Key Components

### 1. Domain-Specific Examples

Each domain has curated few-shot examples:

- **bigquery_examples.json** (3 examples): Web UI queries, code-based querying, schema modification
- **dbt_examples.json** (2 examples): Project initialization, source declaration
- **airbyte_examples.json** (2 examples): Data pipeline setup, data comparison
- **airflow_examples.json** (2 examples): DAG creation with SQL checks, UI navigation
- **dagster_examples.json** (2 examples): Asset definition, materialization
- **excel_examples.json** (2 examples): Formula & chart creation, conditional formatting
- **jupyter_examples.json** (2 examples): Data analysis, notebook creation
- **metabase_examples.json** (2 examples): Visualization creation, dashboard creation
- **servicenow_examples.json** (2 examples): Hardware ordering
- **snowflake_examples.json** (2 examples): Data loading, visualization
- **superset_examples.json** (2 examples): Action log filtering, chart creation

### 2. Domain Detection

Automatic domain detection using keyword scoring:

```python
domain_keywords = {
    "dbt": ["dbt", "snapshot", "seed", "dbt_project", "dbt run"],
    "airbyte": ["airbyte", "faker", "sync now", "connection"],
    "airflow": ["airflow", "dag", "astro", "task"],
    "bigquery": ["bigquery", "bq ", "google cloud"],
    # ... and 7 more domains
}
```

### 3. Task Classification

LLM-based classification into task types:
- `web_ui_query_writing` - UI-based query tasks
- `code_based_querying` - Python/terminal-based tasks
- `schema_modification` - DDL and schema changes
- `data_analysis` - Analysis and visualization
- And more domain-specific types

### 4. Schema Integration (BigQuery)

For BigQuery tasks, integrates with `TablesAnalyzer` to fetch:
- Column names and types
- Sample data rows
- Table descriptions

---

## Generated Instruction Format

All generated instructions follow the Spider2-V format:

```
1. Click the "Destinations" tab on the left navigation bar of the Airbyte UI.
2. Click the row that shows the destination name "Local SQLite".
3. Observe the field labeled "destination_path" in the right-hand panel...
4. Note that Airbyte's Docker image replaces the /local prefix...
5. Open the terminal via clicking the icon on the left menu bar...
6. Type in command "docker ps" to check all running instances...
7. Use docker cp command to copy the database:
   `docker cp ${container_id}:/tmp/airbyte_local/epidemiology.sqlite ~/Desktop/`
```

**Key Features:**
- Numbered steps (1, 2, 3...)
- Action verbs (Click, Type, Navigate, etc.)
- Specific UI elements and paths
- Embedded code snippets with syntax highlighting
- Reasoning annotations ("Note that...", "We will use this later...")
- Collaborative language ("we", "let's")

---

## Design Decisions

### 1. Multi-Domain Architecture
**Choice**: Unified generator with domain-specific customization
**Reasoning**:
- Single codebase easier to maintain
- Shared infrastructure (LLM, prompting, file I/O)
- Domain-specific examples provide specialization
- Easily extensible to new domains

### 2. Few-Shot Learning
**Choice**: Load domain-specific examples from JSON files
**Reasoning**:
- Curated examples ensure high-quality output
- Domain experts can easily update examples
- No need for fine-tuning or training
- Fast iteration and testing

### 3. Directory Structure
**Choice**: Group by domain, then instance ID
**Reasoning**:
- Easier to browse and compare within domains
- Matches spider2-v-extended-dataset structure
- Supports per-domain analysis and evaluation
- Clear organization for 494 examples

### 4. Schema-Aware Generation (BigQuery)
**Choice**: Integrate TablesAnalyzer for BigQuery only
**Reasoning**:
- BigQuery tasks need accurate column names
- Other domains don't have queryable schemas
- Balances accuracy with complexity
- Gracefully degrades if schemas unavailable

---

## Evaluation & Quality

### What Makes a Good Verbose Instruction?

✅ **Completeness**: Covers all steps from start to finish
✅ **Specificity**: Uses actual names, paths, not placeholders
✅ **Clarity**: Each step is actionable and unambiguous
✅ **Domain Expertise**: Follows domain best practices
✅ **Format Consistency**: Matches Spider2-V examples

### Current Performance

Based on testing across all 11 domains:
- ✅ Generates 10-70 detailed steps per task
- ✅ Includes domain-specific terminology
- ✅ Follows numbered format without headers
- ✅ Uses specific UI elements and commands
- ✅ Includes code snippets where appropriate
- ✅ Provides context and reasoning

---

## Limitations & Future Work

### Current Limitations

⚠️ **No Execution Validation**: Generated instructions are not executed
⚠️ **Domain Detection Accuracy**: Keyword-based, may misclassify edge cases
⚠️ **Limited Schema Coverage**: Only BigQuery has schema integration
⚠️ **No Ground Truth Comparison**: Haven't measured similarity to reference instructions

### Future Enhancements

1. **Execution Validation**: Run generated instructions to verify correctness
2. **Improved Domain Detection**: Use LLM-based classification
3. **Schema Integration**: Add schema support for Snowflake, Metabase
4. **Quality Metrics**: BLEU/ROUGE scores vs. ground truth
5. **Iterative Refinement**: LLM self-critique and improvement loop
6. **User Feedback Loop**: Incorporate human feedback for continuous improvement

---

## Files

- `react_verbose_simple.py` - Main generator with multi-domain support
- `utils.py` - TablesAnalyzer for BigQuery schema fetching
- `*_examples.json` - Few-shot examples for each domain (11 files)
- `all_domains_examples.json` - Domain metadata and patterns
- `spider2-v-extended-dataset/` - Input dataset (494 examples)
- `verbose_instructions/` - Generated output (domain-grouped)

---

## Contributing

To add a new domain:

1. Create `{domain}_examples.json` with 2-3 curated examples
2. Add domain keywords to `domain_keywords` in `_detect_domain()`
3. Add domain-specific system prompt to `_build_system_prompt()`
4. Update `all_domains_examples.json` with domain metadata
5. Test with `--per-task-type 1`

---

## Summary

**Goal**: Generate verbose, step-by-step instructions for multi-domain tasks
**Approach**: Multi-domain ReAct agent with few-shot learning using Azure OpenAI
**Coverage**: 11 domains, 494 examples, 23 few-shot examples
**Status**: Fully functional with organized domain-grouped output

This agent provides a comprehensive foundation for automated instruction generation across diverse data analysis and workflow domains.
